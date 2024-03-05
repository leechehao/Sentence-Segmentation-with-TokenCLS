from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class TokenCLSForSentenceSegmentationPipeline:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: Optional[torch.device] = None,
        segmentation_label_id: int = 1,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.model.eval()
        self.segmentation_label_id = segmentation_label_id

    def __call__(self, text: str) -> str:
        results = []
        words = text.split(" ")
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
        )

        torch.cuda.empty_cache()
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            pred_ids = self.model(**inputs).logits.argmax(dim=-1)[0]

        point = 0
        previous_word_idx = 1
        for pred_idx, word_idx in zip(pred_ids, inputs.word_ids()):
            if word_idx is not None and word_idx == previous_word_idx:
                if pred_idx == self.segmentation_label_id:
                    results.append(" ".join(words[point:word_idx]))
                    point = word_idx
                previous_word_idx += 1

        results.append(" ".join(words[point:len(words)]))

        return "\n".join(results)


if __name__ == "__main__":
    pipeline = TokenCLSForSentenceSegmentationPipeline("models/best_model")
    text = "Findings: 1. A 2 cm mass in the right upper lobe, highly suspicious for primary lung cancer. 2. Scattered ground-glass opacities in both lungs, possible early sign of interstitial lung disease. 3. No significant mediastinal lymph node enlargement. 4. Mild pleural effusion on the left side. 5. No evidence of bone metastasis in the visualized portions of the thorax. Conclusion: A. Right upper lobe mass suggestive of lung cancer; biopsy recommended. B. Ground-glass opacities; suggest follow-up CT in 3 months. C. Mild pleural effusion; may require thoracentesis if symptomatic."
    print(pipeline(text))
