"""Model loading utilities for the industrial vision-language pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class IndustrialVLMConfig:
	"""Configuration used to initialize the vision-language model."""

	model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
	trust_remote_code: bool = True
	device_map: Any = "auto"
	torch_dtype: str = "auto"
	load_in_4bit: bool = False
	load_in_8bit: bool = False
	max_new_tokens: int = 256
	temperature: float = 0.2
	top_p: float = 0.9
	system_prompt: str = (
		"You are an industrial inspection assistant. Detect defects, explain likely causes, "
		"and answer in concise professional French unless the user asks otherwise."
	)


def _build_quantization_kwargs(config: IndustrialVLMConfig) -> Dict[str, Any]:
	if not config.load_in_4bit and not config.load_in_8bit:
		return {}

	try:
		from transformers import BitsAndBytesConfig
	except Exception as exc:  # pragma: no cover - optional dependency
		raise RuntimeError(
			"Quantization was requested, but the required transformers/bitsandbytes "
			"dependencies are not available."
		) from exc

	quantization_config = BitsAndBytesConfig(
		load_in_4bit=config.load_in_4bit,
		load_in_8bit=config.load_in_8bit,
	)
	return {"quantization_config": quantization_config}


def load_model_and_processor(config: IndustrialVLMConfig) -> Tuple[Any, Any]:
	"""Load the model and processor with lazy imports.

	The function is intentionally defensive so the package can still be imported on a
	machine where the heavyweight ML stack is not yet installed.
	"""

	try:
		import transformers
	except Exception as exc: 
		raise RuntimeError(
			"Transformers is required to load the industrial VLM model. Install the "
			"dependencies listed in requirements.txt first."
		) from exc

	try:
		processor = transformers.AutoProcessor.from_pretrained(
			config.model_name,
			trust_remote_code=config.trust_remote_code,
		)
	except Exception as exc:  
		raise RuntimeError(f"Unable to load processor for {config.model_name!r}.") from exc

	model_loaders = []
	if hasattr(transformers, "AutoModelForCausalLM"):
		model_loaders.append(transformers.AutoModelForCausalLM)
	if hasattr(transformers, "AutoModelForVision2Seq"):
		model_loaders.append(transformers.AutoModelForVision2Seq)

	last_error: Optional[Exception] = None
	for model_loader in model_loaders:
		try:
			model = model_loader.from_pretrained(
				config.model_name,
				trust_remote_code=config.trust_remote_code,
				device_map=config.device_map,
				torch_dtype=config.torch_dtype,
				**_build_quantization_kwargs(config),
			)
			if hasattr(model, "eval"):
				model.eval()
			return model, processor
		except Exception as exc:  # pragma: no cover - runtime dependency
			last_error = exc

	raise RuntimeError(
		f"Unable to load model {config.model_name!r}. Tried supported auto model classes."
	) from last_error
