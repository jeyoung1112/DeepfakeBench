import logging
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class PixelBranch(nn.Module):

    VARIANTS = {
        "openai/clip-vit-base-patch16": 768,
        "openai/clip-vit-large-patch14": 1024,
    }

    def __init__(self, config):
        super().__init()

        model_name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        lora_rank = config.get('lora_rank', 8)
        lora_alpha = config.get('lora_alpha', 16)
        lora_dropout = config.get('lora_dropout', 0.1)
        lora_targets = config.get('lora_targets', ['q_proj', 'v_proj'])

        clip_model = CLIPVisionModel.from_pretrained(model_name)
        self.vision = clip_model.vision_model
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.out_dim = self.VARIANTS[model_name]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_targets,
            bias="none",
        )
        self.encoder = get_peft_model(self.vision, lora_config)

        for name, param in self.encoder.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        self.norm = nn.LayerNorm(self.out_dim)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"PixelBranch: {trainable:,} trainable / {total:,} total params")

    def forward(self, x):
         outputs = self.encoder(pixel_values=x)
        y_pixel = outputs.pooler_output  # [B, out_dim]
        y_pixel = self.norm(y_pixel)
        return y_pixel