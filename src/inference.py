"""Inference helpers for industrial inspection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from PIL import Image
except Exception:  
    Image = None 

from .model import IndustrialVLMConfig, load_model_and_processor


DEFAULT_INSPECTION_PROMPT = (
    "Inspect this industrial part. Identify visible defects, explain the most likely cause, "
    "and return a concise report with sections labeled Diagnosis, Cause, and Recommended action."
)


@dataclass
class InspectionResult:
    diagnosis: str
    cause: str
    recommendation: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndustrialVLMInspector:
    """High-level inference wrapper around the vision-language model."""

    def __init__(
        self,
        config: Optional[IndustrialVLMConfig] = None,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        self.config = config or IndustrialVLMConfig()
        if model is None or processor is None:
            model, processor = load_model_and_processor(self.config)
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, config: Optional[IndustrialVLMConfig] = None) -> "IndustrialVLMInspector":
        return cls(config=config)

    def _ensure_image(self, image: Union[str, Path, Any]) -> Any:
        if Image is None:
            raise RuntimeError("Pillow is required to load images for inference.")

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image

    def _build_messages(self, image: Any, prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _format_prompt(self, image: Any, prompt: str) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                self._build_messages(image, prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _move_inputs_to_model_device(self, model_inputs: Any) -> Any:
        if hasattr(model_inputs, "to"):
            try:
                return model_inputs.to(self.model.device)
            except Exception:
                return model_inputs
        if isinstance(model_inputs, dict):
            moved_inputs = {}
            for key, value in model_inputs.items():
                moved_inputs[key] = value.to(self.model.device) if hasattr(value, "to") else value
            return moved_inputs
        return model_inputs

    def _decode_output(self, generated_ids: Any) -> str:
        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return decoded[0] if decoded else ""
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
            decoded = self.processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return decoded[0] if decoded else ""
        return str(generated_ids)

    def _extract_field(self, text: str, label: str) -> str:
        pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?:\n[A-Z][A-Za-z ]+\s*:|\Z)"
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def analyze_image(self, image: Union[str, Path, Any], prompt: Optional[str] = None) -> InspectionResult:
        pil_image = self._ensure_image(image)
        user_prompt = prompt or DEFAULT_INSPECTION_PROMPT
        formatted_prompt = self._format_prompt(pil_image, user_prompt)

        if hasattr(self.processor, "__call__"):
            try:
                model_inputs = self.processor(
                    text=[formatted_prompt],
                    images=[pil_image],
                    return_tensors="pt",
                    padding=True,
                )
            except TypeError:
                model_inputs = self.processor(
                    formatted_prompt,
                    images=pil_image,
                    return_tensors="pt",
                )
        else:
            raise RuntimeError("The processor object does not support inference inputs.")

        model_inputs = self._move_inputs_to_model_device(model_inputs)

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        import torch

        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        raw_text = self._decode_output(generated_ids)
        return InspectionResult(
            diagnosis=self._extract_field(raw_text, "Diagnosis") or raw_text,
            cause=self._extract_field(raw_text, "Cause"),
            recommendation=self._extract_field(raw_text, "Recommended action"),
            raw_text=raw_text,
            metadata={"prompt": user_prompt},
        )

    def analyze_directory(self, image_paths: Sequence[Union[str, Path]], prompt: Optional[str] = None) -> List[InspectionResult]:
        return [self.analyze_image(image_path, prompt=prompt) for image_path in image_paths]