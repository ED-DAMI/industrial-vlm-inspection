from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import random_split
from transformers import TrainingArguments
from trl import SFTTrainer

from .dataset import IndustrialDataset
from .model import InspectionModel, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    annotations_file: str = "data/templates/annotations.json"
    image_root: str = "."
    output_dir: str = "checkpoints/vlm-inspection"
    model_mode: str = "quantized_4bit"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    max_seq_length: int = 2048
    val_split: float = 0.1
    seed: int = 42
    use_wandb: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


def train(cfg: TrainConfig) -> None:
    if cfg.use_wandb:
        import wandb
        wandb.init(project="vlm-industrial-inspection", config=vars(cfg))

    model_cfg = ModelConfig(
        mode=cfg.model_mode,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_target_modules=cfg.lora_target_modules,
    )
    inspection_model = InspectionModel(model_cfg).load().apply_lora()
    model = inspection_model.model
    processor = inspection_model.processor

    full_dataset = IndustrialDataset(
        annotations_file=cfg.annotations_file,
        processor=processor,
        image_root=cfg.image_root,
        max_length=cfg.max_seq_length,
    )

    val_size = max(1, int(len(full_dataset) * cfg.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    logger.info("Train: %d | Eval: %d", train_size, val_size)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if cfg.use_wandb else "none",
        seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field=None,
        packing=False,
    )

    logger.info("Démarrage du fine-tuning LoRA...")
    trainer.train()

    output_path = Path(cfg.output_dir) / "final"
    inspection_model.save(output_path)
    logger.info("Fine-tuning terminé. Modèle sauvegardé dans %s", output_path)


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tuning VLM d'inspection industrielle")
    parser.add_argument("--annotations_file", default="data/templates/annotations.json")
    parser.add_argument("--image_root", default=".")
    parser.add_argument("--output_dir", default="checkpoints/vlm-inspection")
    parser.add_argument("--model_mode", default="quantized_4bit",
                        choices=["standard", "quantized_4bit", "quantized_8bit"])
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(_parse_args())