"""Statistiques et visualisation d'un dataset d'images.

Usage :
    python scripts/visualize_data.py data/raw/mvtec --output-dir reports/mvtec

Le nom du dossier parent de chaque image est utilisé comme classe.
Sorties : statistics.json, images.csv, dataset_statistics.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ======================================================================
# VARIABLES À MODIFIER MANUELLEMENT
# ======================================================================
DATASET_DIR = Path("../data/row/mvtec-ad")      # dossier racine des images
OUTPUT_DIR = Path("reports/mvtec")        # dossier de sortie du rapport
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Dossiers ignorés : masques, résultats de modèles, archives.
# Mettre [] pour tout inclure.
DEFAULT_EXCLUDE = ["ground_truth", "results", "reports", ".archives"]
FIGURE_DPI = 130                          # résolution du graphique
FIGURE_SIZE = (18, 10)                    # taille du graphique (pouces)
HIST_BINS = 30                            # nombre de barres des histogrammes
# ======================================================================


def find_images(root: Path, exclude: list[str]) -> list[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTENSIONS or not p.is_file():
            continue
        if any(part in exclude for part in p.relative_to(root).parts):
            continue
        paths.append(p)
    return sorted(paths)


def describe_image(path: Path, root: Path) -> dict:
    rel = path.relative_to(root)
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
        fmt = img.format
        gray = np.asarray(img.convert("L"), dtype=np.float32)
    return {
        "path": rel.as_posix(),
        "class": path.parent.name,
        "top_folder": rel.parts[0] if len(rel.parts) > 1 else "",
        "width": width,
        "height": height,
        "mode": mode,
        "format": fmt,
        "size_kb": round(path.stat().st_size / 1024, 1),
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
    }


def build_stats(df: pd.DataFrame) -> dict:
    return {
        "total_images": int(len(df)),
        "num_classes": int(df["class"].nunique()),
        "images_per_class": df["class"].value_counts().to_dict(),
        "width": df["width"].describe().round(2).to_dict(),
        "height": df["height"].describe().round(2).to_dict(),
        "size_kb": df["size_kb"].describe().round(2).to_dict(),
        "brightness": df["brightness"].describe().round(2).to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "formats": df["format"].value_counts().to_dict(),
        "resolutions": {
            f"{w}x{h}": int(n)
            for (w, h), n in df.groupby(["width", "height"]).size().items()
        },
    }


def plot_stats(df: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
    fig.suptitle(f"Statistiques du dataset ({len(df)} images)", fontsize=15)

    counts = df["class"].value_counts().sort_values()
    counts.plot.barh(ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Images par classe")
    axes[0, 0].set_xlabel("Nombre d'images")

    axes[0, 1].scatter(df["width"], df["height"], alpha=0.4, s=18)
    axes[0, 1].set_title("Dimensions (largeur x hauteur)")
    axes[0, 1].set_xlabel("Largeur (px)")
    axes[0, 1].set_ylabel("Hauteur (px)")

    axes[0, 2].hist(df["size_kb"], bins=HIST_BINS, color="darkorange")
    axes[0, 2].set_title("Taille des fichiers")
    axes[0, 2].set_xlabel("Ko")

    axes[1, 0].hist(df["brightness"], bins=HIST_BINS, color="seagreen")
    axes[1, 0].set_title("Luminosité moyenne")
    axes[1, 0].set_xlabel("Niveau de gris moyen (0-255)")

    axes[1, 1].hist(df["contrast"], bins=HIST_BINS, color="mediumpurple")
    axes[1, 1].set_title("Contraste (écart-type)")
    axes[1, 1].set_xlabel("Écart-type des niveaux de gris")

    modes = df["mode"].value_counts()
    axes[1, 2].pie(modes.values, labels=modes.index, autopct="%1.0f%%")
    axes[1, 2].set_title("Modes couleur")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_file, dpi=FIGURE_DPI)
    plt.close(fig)


def run(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = OUTPUT_DIR,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Analyse le dataset et écrit le rapport. Appelable depuis un notebook."""
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    root = Path(dataset_dir)
    output_dir = Path(output_dir)
    if not root.is_dir():
        raise SystemExit(f"Dossier introuvable : {root}")

    paths = find_images(root, exclude)
    if not paths:
        raise SystemExit(f"Aucune image trouvée dans {root}")
    print(f"{len(paths)} images trouvées, analyse en cours...")

    rows, errors = [], []
    for i, p in enumerate(paths, 1):
        try:
            rows.append(describe_image(p, root))
        except Exception as exc:  # image corrompue
            errors.append({"path": str(p), "error": str(exc)})
        if i % 500 == 0:
            print(f"  {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "images.csv", index=False)
    stats = build_stats(df)
    stats["unreadable_images"] = errors
    (output_dir / "statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_stats(df, output_dir / "dataset_statistics.png")

    print(f"Rapport écrit dans {output_dir}/")
    print(f"  {stats['total_images']} images, {stats['num_classes']} classes")
    if errors:
        print(f"  {len(errors)} image(s) illisible(s), voir statistics.json")
    return df


def main() -> None:
    # Les arguments en ligne de commande sont optionnels :
    # sans argument, les variables du haut du fichier sont utilisées.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    args = parser.parse_args()
    run(args.dataset_dir, args.output_dir, args.exclude)


if __name__ == "__main__":
    main()