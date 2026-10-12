# Assistant Intelligent d'Inspection Industrielle (VLM)

Ce projet implémente un système d'inspection industrielle automatisé basé sur le Vision-Language Model **Qwen2.5-VL**. Contrairement aux systèmes de vision classiques limités à une classification binaire (Bon/Défectueux), cet assistant détecte les anomalies sur des pièces métalliques, les localise, explique leurs causes probables en langage naturel et répond aux questions de l'opérateur en temps réel.

## 🚀 Fonctionnalités
- **Détection & Analyse :** Identification précise des défauts de surface (rayures, fissures, corrosion).
- **Explication Contextuelle :** Génération de rapports explicatifs sur les causes racines (ex: défaut de transport, mauvais réglage machine).
- **Raisonnement Spatial (Grounding) :** Extraction optionnelle des coordonnées de délimitation (bounding boxes) des défauts.
- **Interface Opérateur :** Interface intuitive développée sous Gradio pour une utilisation fluide en usine.

---

## 📂 Structure du Projet

```text
industrial-vlm-inspection/
│
├── data/
│   ├── raw/                  # Datasets d'origine (MVTec AD, NEU)
│   ├── processed/            # Images redimensionnées et masques de défauts
│   └── templates/            # Prompts industriels et annotations (JSON)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Data-loader (Image + Prompt d'instruction)
│   ├── model.py              # Initialisation du modèle Qwen2.5-VL (Quantifié / Standard)
│   ├── train.py              # Script de Fine-Tuning (LoRA / QLoRA)
│   └── inference.py          # Pipeline d'inférence unitaire pour la production
│
├── deployment/
│   ├── app.py                # Interface UI Gradio pour les opérateurs d'usine
│   └── Dockerfile            # Conteneurisation de l'application
│
├── notebooks/                # Notebooks d'expérimentation et d'analyse des données
│   └── 01_data_preparation.ipynb
│
├── requirements.txt          # Dépendances Python requises
└── README.md                 # Documentation du projet
```

## Téléchargement et exploration des données

Après installation des dépendances (`pip install -r requirements.txt`), les
datasets peuvent être téléchargés dans `data/raw/` :

```bash
python scripts/download_data.py --dataset mvtec
python scripts/download_data.py --dataset neu
```

Pour une archive personnalisée :

```powershell
python scripts/download_data.py --url https://example.org/dataset.zip `
  --name mon_dataset --output-dir data/raw
```

Le script évite les téléchargements et extractions déjà réalisés. Utilisez
`--force` pour les refaire. Les archives sont conservées dans
`data/raw/.archives/`.

Les statistiques de base et les graphiques sont générés avec :

```bash
python scripts/visualize_data.py data/raw/mvtec --output-dir reports/mvtec
```

Le rapport contient `statistics.json`, le détail de chaque image dans
`images.csv`, ainsi que `dataset_statistics.png` (répartition des classes et
dimensions des images). Le nom du dossier parent de chaque image est utilisé
comme classe.