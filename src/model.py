from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from .dataset import SYSTEM_PROMPT, build_conversation
from .model import InspectionModel, ModelConfig

logger = logging.getLogger(__name__)

GROUNDING_PROMPT = (
    "Analyse cette pièce et localise chaque défaut avec des bounding boxes. "
    "Réponds au format JSON strict :\n"
    '{"verdict": "CONFORME|NON CONFORME", "defects": ['
    '{"type": "...", "severity": "faible|moyenne|élevée", '
    '"cause": "...", "bbox": [x1, y1, x2, y2]}]}'
)


@dataclass
class InspectionResult:
    verdict: str
    analysis: str
    defects: list[dict[str, Any]]
    inference_time_ms: float
    raw_output: str

    @property
    def is_compliant(self) -> bool:
        return "conforme" in self.verdict.lower() and "non" not in self.verdict.lower()


class InspectionPipeline:
    def __init__(
        self,
        model_path: str | Path | None = None,
        config: ModelConfig | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self._inspection_model = InspectionModel(config).load()
        if model_path:
            self._inspection_model.apply_lora(checkpoint=model_path)
        self.model = self._inspection_model.model
        self.processor = self._inspection_model.processor
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, image: Image.Image, prompt: str) -> tuple[str, float]:
        messages = build_conversation(image, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
        ).to(self.model.device)

        t0 = time.perf_counter()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text, elapsed_ms

    def analyze(self, image: Image.Image, question: str | None = None) -> InspectionResult:
        prompt = question or (
            "Analyse cette pièce métallique. Identifie les défauts, leur type, "
            "localisation et causes probables. Donne un verdict CONFORME ou NON CONFORME."
        )
        raw_output, elapsed_ms = self._generate(image, prompt)

        verdict = _extract_verdict(raw_output)
        return InspectionResult(
            verdict=verdict,
            analysis=raw_output,
            defects=[],
            inference_time_ms=elapsed_ms,
            raw_output=raw_output,
        )

    def analyze_with_grounding(self, image: Image.Image) -> InspectionResult:
        import json

        raw_output, elapsed_ms = self._generate(image, GROUNDING_PROMPT)

        try:
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                verdict = data.get("verdict", _extract_verdict(raw_output))
                defects = data.get("defects", [])
            else:
                verdict = _extract_verdict(raw_output)
                defects = []
        except json.JSONDecodeError:
            logger.warning("Impossible de parser le JSON du grounding.")
            verdict = _extract_verdict(raw_output)
            defects = []

        return InspectionResult(
            verdict=verdict,
            analysis=raw_output,
            defects=defects,
            inference_time_ms=elapsed_ms,
            raw_output=raw_output,
        )

    def chat(
        self,
        image: Image.Image,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        context = ""
        if history:
            for q, a in history[-3:]:
                context += f"Opérateur: {q}\nAssistant: {a}\n\n"
        full_prompt = f"{context}Opérateur: {question}" if context else question
        output, _ = self._generate(image, full_prompt)
        return output


def _extract_verdict(text: str) -> str:
    upper = text.upper()
    if "NON CONFORME" in upper:
        return "NON CONFORME"
    if "CONFORME" in upper:
        return "CONFORME"
    return "INDÉTERMINÉ"