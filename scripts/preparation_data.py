"""Prétraitement de MVTec AD pour le fine-tuning du VLM.

Étapes :
  1. lecture de la taxonomie (configs/defect_classes.yaml)
  2. collecte des images (train/good + test/*) et des masques (ground_truth/*)
  3. sous-échantillonnage (seedé) de la classe « good »
  4. redimensionnement carré en PNG RGB -> data/processed/images/
     masques redimensionnés (NEAREST)      -> data/processed/masks/
  5. boîtes englobantes calculées depuis les masques
  6. paires question/réponse (français)
  7. split train/val/test stratifié avec graine fixe
  8. écriture de data/templates/{train,val,test}.json et manifest.csv

Usage : modifier les variables ci-dessous, puis `python scripts/prepare_data.py`
ou, depuis un notebook : `import prepare_data as p; p.run()`.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

# ======================================================================
# VARIABLES À MODIFIER MANUELLEMENT
# ======================================================================
MVTEC_DIR = Path("../data/row/mvtec-ad")           # contient bottle/, cable/, ...
PROCESSED_DIR = Path("../data/processed")       # images/ et masks/ y seront créés
TEMPLATES_DIR = Path("../data/templates")       # train.json, val.json, test.json
TAXONOMY_FILE = Path("../configs/defect_classes.yaml")

CATEGORIES = None        # None = toutes ; ou ["metal_nut", "screw", "transistor"]
IMAGE_SIZE = 512         # côté de l'image traitée (carré)
SEED = 42                # graine unique : sous-échantillonnage, splits, questions
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
GOOD_RATIO = 2.0         # max d'images "good" = GOOD_RATIO x nb de défectueuses
                         # par catégorie. None = garder toutes les images good.
MIN_BOX_AREA = 16        # pixels (image traitée) en dessous desquels une
                         # composante du masque est ignorée
OVERWRITE = False        # True = retraiter les images déjà présentes
# ======================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

QUESTIONS = [
    "Cette pièce présente-t-elle un défaut ?",
    "Inspecte cette pièce : y a-t-il une anomalie ?",
    "La pièce est-elle conforme ? Si non, décris le défaut et sa position.",
    "Analyse l'image de cette pièce et indique tout défaut visible.",
]


# ----------------------------------------------------------------------
# Taxonomie
# ----------------------------------------------------------------------
def load_taxonomy(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = {c["name"]: c for c in data["classes"]}
    source_map: dict[str, str] = {}
    for c in data["classes"]:
        for src in c.get("sources", {}).get("mvtec", []) or []:
            if src in source_map:
                raise ValueError(
                    f"Défaut « {src} » présent dans « {source_map[src]} » et « {c['name']} »"
                )
            source_map[src] = c["name"]
    if "good" not in classes:
        raise ValueError("La classe « good » est obligatoire dans la taxonomie")
    return {
        "classes": classes,
        "source_map": source_map,
        "categories": data.get("categories", {}),
    }


# ----------------------------------------------------------------------
# Collecte
# ----------------------------------------------------------------------
def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_samples(root: Path, categories: list[str] | None, tax: dict) -> list[dict]:
    if not root.is_dir():
        raise SystemExit(f"Dossier MVTec introuvable : {root}")
    cats = categories or sorted(
        d.name for d in root.iterdir() if (d / "train").is_dir() and (d / "test").is_dir()
    )
    if not cats:
        raise SystemExit(f"Aucune catégorie MVTec (train/ + test/) dans {root}")

    samples, unknown = [], set()
    for cat in cats:
        for orig_split in ("train", "test"):
            split_dir = root / cat / orig_split
            if not split_dir.is_dir():
                continue
            for defect_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
                defect = defect_dir.name
                if defect == "good":
                    cls = "good"
                elif defect in tax["source_map"]:
                    cls = tax["source_map"][defect]
                else:
                    unknown.add(defect)
                    continue
                for img in list_images(defect_dir):
                    mask = None
                    if cls != "good":
                        m = root / cat / "ground_truth" / defect / f"{img.stem}_mask.png"
                        mask = m if m.is_file() else None
                    samples.append(
                        {
                            "category": cat,
                            "original_defect": defect,
                            "original_split": orig_split,
                            "defect_class": cls,
                            "src_image": img,
                            "src_mask": mask,
                        }
                    )
    if unknown:
        raise SystemExit(
            "Défauts absents de la taxonomie : "
            f"{sorted(unknown)}. Ajoutez-les dans {TAXONOMY_FILE}."
        )
    return samples


def subsample_good(samples: list[dict], ratio: float | None, rng: random.Random) -> list[dict]:
    if ratio is None:
        return samples
    by_cat = defaultdict(lambda: {"good": [], "bad": []})
    for s in samples:
        by_cat[s["category"]]["good" if s["defect_class"] == "good" else "bad"].append(s)
    kept = []
    for cat in sorted(by_cat):
        good, bad = by_cat[cat]["good"], by_cat[cat]["bad"]
        n_keep = min(len(good), math.ceil(ratio * len(bad)))
        good = sorted(good, key=lambda s: str(s["src_image"]))
        kept += rng.sample(good, n_keep) + bad
    return kept


def assign_ids(samples: list[dict]) -> None:
    counters: Counter = Counter()
    samples.sort(
        key=lambda s: (s["category"], s["original_defect"], s["original_split"], s["src_image"].name)
    )
    for s in samples:
        key = (s["category"], s["original_defect"])
        s["id"] = f"mvtec_{s['category']}_{s['original_defect']}_{counters[key]:04d}"
        counters[key] += 1


# ----------------------------------------------------------------------
# Traitement image / masque
# ----------------------------------------------------------------------
def mask_to_boxes(mask: np.ndarray, min_area: int) -> list[list[int]]:
    """Composantes connexes (8-connexité) -> [x0, y0, x1, y1], x1/y1 exclusifs."""
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labeled), start=1):
        if sl is None or int((labeled[sl] == i).sum()) < min_area:
            continue
        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
    return boxes


def process_sample(s: dict, img_dir: Path, mask_dir: Path, size: int, overwrite: bool) -> None:
    out_img = img_dir / f"{s['id']}.png"
    out_mask = mask_dir / f"{s['id']}.png"

    with Image.open(s["src_image"]) as img:
        s["orig_size"] = list(img.size)
        if overwrite or not out_img.exists():
            img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(out_img)
    s["image_path"] = out_img.as_posix()

    s["bboxes"], s["mask_path"] = [], None
    if s["src_mask"] is not None:
        with Image.open(s["src_mask"]) as m:
            m = m.convert("L").resize((size, size), Image.Resampling.NEAREST)
        arr = np.asarray(m) > 0
        if overwrite or not out_mask.exists():
            Image.fromarray((arr * 255).astype(np.uint8)).save(out_mask)
        s["mask_path"] = out_mask.as_posix()
        s["bboxes"] = mask_to_boxes(arr, MIN_BOX_AREA)


# ----------------------------------------------------------------------
# Annotations question/réponse
# ----------------------------------------------------------------------
def describe_position(box: list[int], size: int) -> str:
    cx, cy = (box[0] + box[2]) / 2 / size, (box[1] + box[3]) / 2 / size
    v = "en haut" if cy < 1 / 3 else "en bas" if cy > 2 / 3 else ""
    h = "à gauche" if cx < 1 / 3 else "à droite" if cx > 2 / 3 else ""
    return " ".join(p for p in (v, h) if p) or "au centre de l'image"


def make_qa(s: dict, tax: dict, size: int) -> tuple[str, str]:
    rng = random.Random(f"{SEED}:{s['id']}")
    question = rng.choice(QUESTIONS)
    cat_fr = tax["categories"].get(s["category"], s["category"].replace("_", " "))
    if s["defect_class"] == "good":
        return question, f"Conforme. La pièce ({cat_fr}) ne présente aucun défaut visible."

    label = tax["classes"][s["defect_class"]]["label_fr"]
    boxes = s["bboxes"]
    if not boxes:
        return question, f"Non conforme. La pièce ({cat_fr}) présente un défaut : {label}."
    biggest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    where = describe_position(biggest, size)
    if len(boxes) == 1:
        return question, (
            f"Non conforme. La pièce ({cat_fr}) présente un défaut : {label}, situé {where}."
        )
    return question, (
        f"Non conforme. La pièce ({cat_fr}) présente un défaut : {label}, "
        f"réparti sur {len(boxes)} zones, la plus grande étant située {where}."
    )


# ----------------------------------------------------------------------
# Split stratifié
# ----------------------------------------------------------------------
def assign_splits(records: list[dict], ratios: dict, seed: int) -> list[str]:
    """Stratifié par (catégorie, classe). Retourne la liste des groupes trop petits."""
    if abs(sum(ratios.values()) - 1) > 1e-6:
        raise ValueError("SPLIT_RATIOS doit sommer à 1")
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r["id"]):
        groups[(r["category"], r["defect_class"])].append(r)

    too_small = []
    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            n_val = n_test = 0
            too_small.append(f"{key[0]}/{key[1]} ({n})")
        else:
            n_val = max(1, round(n * ratios["val"]))
            n_test = max(1, round(n * ratios["test"]))
            if n_val + n_test >= n:         # garder au moins 1 image en train
                n_val = n_test = 1
        for i, r in enumerate(items):
            r["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return too_small


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run() -> dict[str, list[dict]]:
    tax = load_taxonomy(TAXONOMY_FILE)
    rng = random.Random(SEED)

    samples = collect_samples(MVTEC_DIR, CATEGORIES, tax)
    n_raw = len(samples)
    samples = subsample_good(samples, GOOD_RATIO, rng)
    assign_ids(samples)
    print(f"{n_raw} images collectées, {len(samples)} conservées après sous-échantillonnage.")

    img_dir, mask_dir = PROCESSED_DIR / "images", PROCESSED_DIR / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for i, s in enumerate(samples, 1):
        process_sample(s, img_dir, mask_dir, IMAGE_SIZE, OVERWRITE)
        question, answer = make_qa(s, tax, IMAGE_SIZE)
        records.append(
            {
                "id": s["id"],
                "image_path": s["image_path"],
                "mask_path": s["mask_path"],
                "dataset": "mvtec",
                "category": s["category"],
                "original_defect": s["original_defect"],
                "defect_class": s["defect_class"],
                "defect_class_id": tax["classes"][s["defect_class"]]["id"],
                "is_defective": s["defect_class"] != "good",
                "bboxes": s["bboxes"],
                "image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "original_size": s["orig_size"],
                "question": question,
                "answer": answer,
                "_original_path": s["src_image"].as_posix(),
            }
        )
        if i % 500 == 0:
            print(f"  {i}/{len(samples)} images traitées")

    too_small = assign_splits(records, SPLIT_RATIOS, SEED)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split"]].append({k: v for k, v in r.items() if k != "_original_path"})
    # Anti-fuite : un identifiant ne doit apparaître que dans un seul split.
    ids = [r["id"] for part in splits.values() for r in part]
    assert len(ids) == len(set(ids)), "Identifiants dupliqués entre splits"

    for name, part in splits.items():
        (TEMPLATES_DIR / f"{name}.json").write_text(
            json.dumps(part, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    with open(PROCESSED_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "original_path", "image_path", "split", "category",
                    "original_defect", "defect_class"])
        for r in records:
            w.writerow([r["id"], r["_original_path"], r["image_path"], r["split"],
                        r["category"], r["original_defect"], r["defect_class"]])

    # Résumé
    print("\nRépartition :")
    for name, part in splits.items():
        n_bad = sum(r["is_defective"] for r in part)
        print(f"  {name:<5} {len(part):>5} images  ({n_bad} défectueuses)")
    no_box = sum(1 for r in records if r["is_defective"] and not r["bboxes"])
    if no_box:
        print(f"\nAttention : {no_box} images défectueuses sans boîte (masque absent ou vide).")
    if too_small:
        print("\nGroupes < 3 images, placés entièrement en train :")
        print("  " + ", ".join(too_small))
    print(f"\nAnnotations : {TEMPLATES_DIR}/  |  images : {img_dir}/")
    return splits


if __name__ == "__main__":
    run()