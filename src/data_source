from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

DEFECT_LABELS = [
    "crainte", "rayure", "fissure", "piqûre",
    "corrosion", "inclusion", "patch", "rouille",
]

SYSTEM_PROMPT = (
    "Tu es un expert en contrôle qualité industriel. "
    "Tu analyses des images de pièces métalliques pour détecter les défauts de surface. "
    "Pour chaque image, tu identifies les anomalies, leur localisation, leur sévérité "
    "et leurs causes probables. Tu réponds en français, de manière précise et structurée."
)

DEFAULT_USER_PROMPT = (
    "Analyse cette pièce métallique. "
    "Y a-t-il des défauts visibles ? Si oui, décris leur type, leur localisation "
    "et leurs causes probables. Donne un verdict final : CONFORME ou NON CONFORME."
)


def build_conversation(
    image: Image.Image,
    user_prompt: str = DEFAULT_USER_PROMPT,
    assistant_response: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    if assistant_response:
        messages.append({"role": "assistant", "content": assistant_response})
    return messages


class IndustrialDataset(Dataset):
    """
    Dataset pour le fine-tuning du modèle d'inspection industrielle.

    Chaque entrée JSON dans `annotations_file` doit avoir la structure :
    {
        "image_path": "data/processed/metal_nut/scratch_001.png",
        "user_prompt": "...",       (optionnel)
        "label": "scratch",
        "response": "..."
    }
    """

    def __init__(
        self,
        annotations_file: str | Path,
        processor: AutoProcessor,
        image_root: str | Path = ".",
        max_length: int = 2048,
    ) -> None:
        self.processor = processor
        self.image_root = Path(image_root)
        self.max_length = max_length

        with open(annotations_file, encoding="utf-8") as f:
            self.samples: list[dict[str, Any]] = json.load(f)

        logger.info("Dataset chargé : %d échantillons", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.image_root / sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        user_prompt = sample.get("user_prompt", DEFAULT_USER_PROMPT)
        response = sample["response"]

        messages = build_conversation(image, user_prompt, response)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def generate_annotation_template(
    processed_dir: str | Path,
    output_path: str | Path,
) -> None:
    processed_dir = Path(processed_dir)
    templates: list[dict[str, Any]] = []

    for img_path in sorted(processed_dir.rglob("*.png")):
        defect_type = img_path.parent.name
        templates.append(
            {
                "image_path": str(img_path.relative_to(processed_dir.parent)),
                "label": defect_type,
                "user_prompt": DEFAULT_USER_PROMPT,
                "response": (
                    f"**Analyse de la pièce :**\n\n"
                    f"- **Défaut détecté :** {defect_type}\n"
                    f"- **Localisation :** À compléter\n"
                    f"- **Sévérité :** À compléter (Faible / Moyenne / Élevée)\n"
                    f"- **Cause probable :** À compléter\n\n"
                    f"**Verdict : NON CONFORME**"
                ),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

    logger.info("Template d'annotations généré : %s (%d entrées)", output_path, len(templates))