"""Compute and visualize basic statistics for an image dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_statistics(data_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
                records.append(
                    {
                        "path": str(path.relative_to(data_dir)),
                        "label": path.parent.name,
                        "width": width,
                        "height": height,
                        "mode": image.mode,
                        "format": image.format or path.suffix.lower().lstrip("."),
                    }
                )
        except (OSError, UnidentifiedImageError) as error:
            errors.append(f"{path}: {error}")
    return records, errors


def write_report(records: list[dict[str, object]], errors: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = Counter(str(record["label"]) for record in records)
    widths = [int(record["width"]) for record in records]
    heights = [int(record["height"]) for record in records]
    report = {
        "image_count": len(records),
        "invalid_image_count": len(errors),
        "class_counts": dict(sorted(labels.items())),
        "width": {"min": min(widths), "max": max(widths), "mean": sum(widths) / len(widths)}
        if widths
        else {},
        "height": {"min": min(heights), "max": max(heights), "mean": sum(heights) / len(heights)}
        if heights
        else {},
        "errors": errors,
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "images.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "width", "height", "mode", "format"])
        writer.writeheader()
        writer.writerows(records)


def create_plots(records: list[dict[str, object]], output_dir: Path) -> None:
    if not records:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("La visualisation nécessite matplotlib (pip install matplotlib).") from error

    labels = Counter(str(record["label"]) for record in records)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels.keys(), labels.values())
    axes[0].set_title("Images par classe")
    axes[0].set_xlabel("Classe")
    axes[0].set_ylabel("Nombre d'images")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].scatter(
        [int(record["width"]) for record in records],
        [int(record["height"]) for record in records],
        alpha=0.6,
    )
    axes[1].set_title("Dimensions des images")
    axes[1].set_xlabel("Largeur (px)")
    axes[1].set_ylabel("Hauteur (px)")
    figure.tight_layout()
    figure.savefig(output_dir / "dataset_statistics.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Dossier contenant les images")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Dossier introuvable : {args.data_dir}")
    records, errors = collect_statistics(args.data_dir)
    write_report(records, errors, args.output_dir)
    create_plots(records, args.output_dir)
    print(f"{len(records)} image(s) analysée(s), {len(errors)} erreur(s).")
    print(f"Rapport : {args.output_dir / 'statistics.json'}")


if __name__ == "__main__":
    main()
