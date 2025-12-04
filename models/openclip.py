import torch
import torch.nn as nn
from transformers import CLIPTextModel, AutoTokenizer

class ClipGmPWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        model_name = "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        self.device = device
        self.max_length = 77  # typical CLIP token length

    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # return per-token hidden states [B, 77, 768]
        return out.last_hidden_state
