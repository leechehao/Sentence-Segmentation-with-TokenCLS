import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import datasets

import utils

TEST = "test"
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"
NER_TAGS = "ner_tags"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True, help="The path to a file containing the test data.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="The name or path of a pretrained model that the script should use.")
    parser.add_argument("--cache_dir", default="cache_dir", type=str)
    parser.add_argument("--max_length", default=512, type=int, help="The maximum length of the input sequences.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="The batch size to use during evaluation.")
    args = parser.parse_args()

    # ===== Load file =====
    dataset = datasets.load_dataset(
        path=args.script_file,
        data_files={
            TEST: args.test_file,
        },
        cache_dir=args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    encode_with_label = partial(utils.encode_with_label, tokenizer=tokenizer, max_length=args.max_length)
    dataset = dataset.map(encode_with_label, batched=True)
    dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    test_dataloader = DataLoader(dataset[TEST], batch_size=args.eval_batch_size)
    label_list = dataset[TEST].features[NER_TAGS].feature.names

    # ===== Model =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_name_or_path).to(device)

    test_loss, test_f1_sos, test_f1_eos, test_cls_report, test_duration = utils.evaluate(model, test_dataloader, device, label_list)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1 (SOS): {test_f1_sos:.4f}")
    print(f"Test F1 (EOS): {test_f1_eos:.4f}")
    print(f"Test CLS report:\n{test_cls_report}")
    print(f"Test Duration: {test_duration:.3f} sec")


if __name__ == "__main__":
    main()
