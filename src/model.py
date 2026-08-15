from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass
class ModelConfig:
    model_id: str = MODEL_ID
    mode: Literal["standard", "quantized_4bit", "quantized_8bit"] = "quantized_4bit"
    device: str = "auto"
    max_pixels: int = 1_280 * 28 * 28
    min_pixels: int = 256 * 28 * 28
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


def _build_bnb_config(mode: str) -> BitsAndBytesConfig | None:
    if mode == "quantized_4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if mode == "quantized_8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class InspectionModel:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.processor: AutoProcessor | None = None
        self.model: Qwen2_5_VLForConditionalGeneration | None = None

    def load(self) -> "InspectionModel":
        cfg = self.config
        logger.info("Chargement du processeur depuis %s", cfg.model_id)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
        )

        bnb_config = _build_bnb_config(cfg.mode)
        torch_dtype = torch.bfloat16 if cfg.mode == "standard" else None

        logger.info("Chargement du modèle (mode=%s)", cfg.mode)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            device_map=cfg.device,
            quantization_config=bnb_config,
        )
        logger.info("Modèle chargé avec succès.")
        return self

    def apply_lora(self, checkpoint: str | Path | None = None) -> "InspectionModel":
        assert self.model is not None, "Appeler .load() avant .apply_lora()"
        if checkpoint:
            logger.info("Chargement des poids LoRA depuis %s", checkpoint)
            self.model = PeftModel.from_pretrained(self.model, str(checkpoint))
        else:
            logger.info("Initialisation d'un nouvel adaptateur LoRA")
            lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()
        return self

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        self.processor.save_pretrained(str(output_dir))
        logger.info("Modèle sauvegardé dans %s", output_dir)