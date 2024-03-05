import os
import time
import argparse
from pathlib import Path
from functools import partial

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import datasets
from transformers import set_seed, AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
import mlflow
from mlflow.models.signature import infer_signature

import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENS = "tokens"
MAX_LENGTH = "max_length"
NER_TAGS = "ner_tags"
IGNORE_INDEX = -100
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_path", type=str, required=True, help="")
    parser.add_argument("--experiment_name", type=str, required=True, help="")
    parser.add_argument("--run_name", type=str, required=True, help="")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--script_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--log_file", default="train.log", type=str, help="")
    parser.add_argument("--cache_dir", default="cache_dir", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default="prajjwal1/bert-tiny", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--warmup_ratio", default=0.0, type=float)
    parser.add_argument("--accum_steps", default=1, type=int)
    parser.add_argument("--max_norm", default=1.0, type=float)
    parser.add_argument("--seed", default=2330, type=int)
    args = parser.parse_args()

    # ===== Set seed =====
    set_seed(args.seed)

    # ===== Set tracking URI =====
    EXPERIMENTS_PATH = Path(args.experiments_path)
    EXPERIMENTS_PATH.mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri(EXPERIMENTS_PATH)

    # ===== Set experiment =====
    mlflow.set_experiment(experiment_name=args.experiment_name)

    # ===== Load file =====
    dataset = datasets.load_dataset(
        path=args.script_file,
        data_files={
            TRAIN: args.train_file,
            VALIDATION: args.validation_file,
            TEST: args.test_file,
        },
        cache_dir=args.cache_dir,
    )
    log_file = open(args.log_file, "w", encoding="utf-8")

    # ===== Preprocessing =====
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.save_pretrained(args.model_path)

    encode_with_label_max = partial(
        utils.encode_with_label,
        tokenizer=tokenizer,
        padding=MAX_LENGTH,
        max_length=args.max_length,
    )
    encode_with_label = partial(
        utils.encode_with_label,
        tokenizer=tokenizer,
    )
    dataset[TRAIN] = dataset[TRAIN].map(encode_with_label_max, batched=True)
    dataset[VALIDATION] = dataset[VALIDATION].map(encode_with_label, batched=True)
    dataset[TEST] = dataset[TEST].map(encode_with_label, batched=True)
    dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    train_dataloader = DataLoader(dataset[TRAIN], batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset[VALIDATION], batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset[TEST], batch_size=args.batch_size)

    # ===== Model =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list = dataset[TRAIN].features[NER_TAGS].feature.names
    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=len(label_list),
    ).to(device)

    # ===== Optimizer ======
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    ###################################
    ##########     Train     ##########
    ###################################

    # ===== Tracking =====
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(vars(args))
        best_val_f1 = 0
        epochs = tqdm(range(args.epochs), desc="Epoch ... ", position=0)
        for epoch in epochs:
            model.train()
            train_losses = utils.AverageMeter()
            start_time = time.time()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training ... ", position=1)):
                batch_size = batch[LABELS].size(0)
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                outputs = model(**batch)
                loss = outputs.loss
                train_losses.update(loss.item(), batch_size)

                if args.accum_steps > 1:
                    loss = loss / args.accum_steps
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

                if (step + 1) % args.accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if (step + 1) == 1 or (step + 1) % 2500 == 0 or (step + 1) == len(train_dataloader):
                    epochs.write(
                        f"Epoch: [{epoch + 1}][{step + 1}/{len(train_dataloader)}] "
                        f"Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) "
                        f"Grad: {grad_norm:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.8f}",
                        file=log_file,
                    )
                    log_file.flush()
                    os.fsync(log_file.fileno())

                    mlflow.log_metrics(
                        {
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                        step=(len(train_dataloader) * epoch) + step,
                    )
            train_duration = time.time() - start_time
            epochs.write(f"Training Duration: {train_duration:.3f} sec", file=log_file)

            ####################################
            ##########   Validation   ##########
            ####################################

            val_loss, val_f1_sos, val_f1_eos, val_cls_report, val_duration = utils.evaluate(
                model,
                val_dataloader,
                device,
                label_list,
            )
            epochs.write(f"Validation Loss: {val_loss:.4f}", file=log_file)
            epochs.write(f"Validation F1 (SOS): {val_f1_sos:.4f}", file=log_file)
            epochs.write(f"Validation F1 (EOS): {val_f1_eos:.4f}", file=log_file)
            epochs.write(f"Validation CLS report:\n{val_cls_report}", file=log_file)
            epochs.write(f"Validation Duration: {val_duration:.3f} sec\n", file=log_file)

            if val_f1_sos > best_val_f1:
                model.save_pretrained(args.model_path)
                best_val_f1 = val_f1_sos

            mlflow.log_metrics(
                {
                    "train_loss": train_losses.avg,
                    "val_loss": val_loss,
                    "val_f1_sos": val_f1_sos,
                    "val_f1_eos": val_f1_eos,
                    "best_val_f1": best_val_f1,
                },
                step=epoch,
            )

        ####################################
        ##########      Test      ##########
        ####################################

        model = model.from_pretrained(args.model_path).to(device)
        test_loss, test_f1_sos, test_f1_eos, test_cls_report, test_duration = utils.evaluate(
            model,
            test_dataloader,
            device,
            label_list,
        )
        epochs.write(f"Test Loss: {test_loss:.4f}", file=log_file)
        epochs.write(f"Test F1 (SOS): {test_f1_sos:.4f}", file=log_file)
        epochs.write(f"Test F1 (EOS): {test_f1_eos:.4f}", file=log_file)
        epochs.write(f"Test CLS report:\n{test_cls_report}", file=log_file)
        epochs.write(f"Test Duration: {test_duration:.3f} sec", file=log_file)

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_f1_sos": test_f1_sos,
                "test_f1_eos": test_f1_eos,
            },
        )

        log_file.close()
        mlflow.log_artifact(args.log_file)

        # ===== Package model file to mlflow =====
        artifacts = {
            Path(file).stem: os.path.join(args.model_path, file)
            for file in os.listdir(args.model_path)
            if not os.path.basename(file).startswith('.')
        }

        sample = pd.DataFrame({"text": ["nodule in right upper lung . mass in left lower lung ."]})
        results = pd.DataFrame({"results": ["nodule in right upper lung .\nmass in left lower lung ."]})
        signature = infer_signature(sample, results)

        segmentation_label_id = dataset[TRAIN].features[NER_TAGS].feature.names.index("B-S")

        mlflow.pyfunc.log_model(
            "model",
            python_model=utils.TokenCLSForSentenceSegmentationAPI(segmentation_label_id),
            code_path=["utils.py"],
            artifacts=artifacts,
            signature=signature,
        )


if __name__ == "__main__":
    main()
