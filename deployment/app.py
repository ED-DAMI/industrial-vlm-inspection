from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import InspectionPipeline
from src.model import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESCRIPTION = """
# 🔍 Assistant Intelligent d'Inspection Industrielle

Modèle : **Qwen2.5-VL** | Fine-tuné sur données MVTec AD & NEU
Soumettre une image de pièce métallique pour analyse automatique des défauts.
"""

CSS = """
.verdict-ok   { color: #16a34a; font-weight: bold; font-size: 1.2rem; }
.verdict-nok  { color: #dc2626; font-weight: bold; font-size: 1.2rem; }
.verdict-unk  { color: #d97706; font-weight: bold; font-size: 1.2rem; }
footer { display: none !important; }
"""

_pipeline: InspectionPipeline | None = None


def _get_pipeline() -> InspectionPipeline:
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline non initialisé. Lancer load_pipeline() d'abord.")
    return _pipeline


def load_pipeline(model_path: str | None, mode: str) -> None:
    global _pipeline
    config = ModelConfig(mode=mode)
    _pipeline = InspectionPipeline(
        model_path=model_path if model_path else None,
        config=config,
    )
    logger.info("Pipeline chargé avec succès.")


def analyze_image(
    image: Image.Image | None,
    with_grounding: bool,
) -> tuple[str, str]:
    if image is None:
        return "⚠️ Aucune image fournie.", ""
    pipeline = _get_pipeline()
    if with_grounding:
        result = pipeline.analyze_with_grounding(image)
    else:
        result = pipeline.analyze(image)

    verdict_html = _format_verdict(result.verdict)
    analysis_md = (
        f"**Temps d'inférence :** {result.inference_time_ms:.0f} ms\n\n"
        f"---\n\n{result.analysis}"
    )
    if result.defects:
        analysis_md += "\n\n---\n\n**Défauts localisés (JSON) :**\n```json\n"
        import json
        analysis_md += json.dumps(result.defects, ensure_ascii=False, indent=2)
        analysis_md += "\n```"

    return verdict_html, analysis_md


def chat_with_image(
    image: Image.Image | None,
    message: str,
    history: list[list[str]],
) -> tuple[list[list[str]], str]:
    if image is None:
        history.append([message, "⚠️ Veuillez d'abord soumettre une image."])
        return history, ""
    if not message.strip():
        return history, ""

    pipeline = _get_pipeline()
    formatted_history = [(h[0], h[1]) for h in history]
    response = pipeline.chat(image, message, formatted_history)

    history.append([message, response])
    return history, ""


def _format_verdict(verdict: str) -> str:
    if "NON CONFORME" in verdict.upper():
        return f'<div class="verdict-nok">❌ {verdict}</div>'
    if "CONFORME" in verdict.upper():
        return f'<div class="verdict-ok">✅ {verdict}</div>'
    return f'<div class="verdict-unk">⚠️ {verdict}</div>'


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Inspection Industrielle VLM") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Tabs():
            with gr.TabItem("🔎 Analyse Automatique"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Image de la pièce",
                            height=400,
                        )
                        with_grounding = gr.Checkbox(
                            label="Activer le grounding (bounding boxes)",
                            value=False,
                        )
                        analyze_btn = gr.Button("Analyser", variant="primary")
                    with gr.Column(scale=1):
                        verdict_output = gr.HTML(label="Verdict")
                        analysis_output = gr.Markdown(label="Rapport d'analyse")

                gr.Examples(
                    examples=[
                        ["data/raw/metal_nut/scratch_001.png", False],
                        ["data/raw/metal_nut/crack_002.png", True],
                    ],
                    inputs=[image_input, with_grounding],
                    label="Exemples de pièces",
                )

                analyze_btn.click(
                    fn=analyze_image,
                    inputs=[image_input, with_grounding],
                    outputs=[verdict_output, analysis_output],
                )

            with gr.TabItem("💬 Dialogue Opérateur"):
                gr.Markdown(
                    "Posez des questions spécifiques sur la pièce après avoir soumis une image."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        chat_image = gr.Image(
                            type="pil",
                            label="Image de référence",
                            height=350,
                        )
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(height=400, label="Conversation")
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Ex: La fissure est-elle critique ? Peut-on utiliser cette pièce ?",
                                label="Votre question",
                                scale=4,
                            )
                            send_btn = gr.Button("Envoyer", scale=1, variant="primary")
                        clear_btn = gr.Button("Effacer la conversation")

                send_btn.click(
                    fn=chat_with_image,
                    inputs=[chat_image, chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                chat_input.submit(
                    fn=chat_with_image,
                    inputs=[chat_image, chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Interface d'inspection industrielle VLM")
    parser.add_argument("--model_path", default=None, help="Chemin vers les poids LoRA fine-tunés")
    parser.add_argument("--mode", default="quantized_4bit",
                        choices=["standard", "quantized_4bit", "quantized_8bit"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_pipeline(args.model_path, args.mode)
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()