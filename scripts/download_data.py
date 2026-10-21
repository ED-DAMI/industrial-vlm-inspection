
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import snapshot_download


# ==================================================
# 1. CONFIGURATION DES CHEMINS
# ==================================================

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "mvtec-ad"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. TÉLÉCHARGEMENT DU DATASET
# ==================================================

REPO_ID = "foersben/mvtec-ad"

path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(DATA_DIR)
)

print(f"Dataset téléchargé dans : {path}")


# ==================================================
# 3. CHARGEMENT DU DATASET
# ==================================================

ds = load_dataset(REPO_ID)

print(ds)


# ==================================================
# 4. AFFICHAGE D'UNE IMAGE EXEMPLE
# ==================================================

example = ds["train"][0]
image = example["image"]

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Exemple d'image du dataset MVTec AD")
plt.axis("off")
plt.tight_layout()
plt.show()
