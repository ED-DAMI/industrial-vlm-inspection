"""Download and extract the datasets used by the inspection project.

Examples:
    python scripts/download_data.py --dataset mvtec
    python scripts/download_data.py --url https://example.org/dataset.zip --name custom
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_URLS = {
    "mvtec": (
        "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/products/"
        "datasets/2019_05_16_downloads/mvtec_anomaly_detection.tar.xz"
    ),
    "neu": "https://github.com/abin24/Surface-Defect-Detection/archive/refs/heads/master.zip",
}


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as files:
        for member in files.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        files.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as files:
        for member in files.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.name}")
        files.extractall(destination)


def download_dataset(
    name: str,
    url: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download one archive and extract it below ``output_dir/name``."""
    dataset_dir = output_dir / name
    archive_dir = output_dir / ".archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".zip" if url.lower().split("?")[0].endswith(".zip") else ".tar.xz"
    archive = archive_dir / f"{name}{suffix}"
    if force or not archive.exists():
        print(f"Téléchargement de {name} depuis {url}")
        urllib.request.urlretrieve(url, archive)
    else:
        print(f"Archive existante réutilisée : {archive}")

    marker = dataset_dir / ".extracted"
    if force or not marker.exists():
        if force and dataset_dir.exists():
            for child in dataset_dir.iterdir():
                if child.name != ".extracted":
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
        if suffix == ".zip":
            _safe_extract_zip(archive, dataset_dir)
        else:
            _safe_extract_tar(archive, dataset_dir)
        marker.touch()
        print(f"Données extraites dans {dataset_dir}")
    else:
        print(f"Données déjà extraites dans {dataset_dir}")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_URLS), default="mvtec")
    parser.add_argument("--url", help="URL d'une archive personnalisée")
    parser.add_argument("--name", help="Nom du dossier pour une URL personnalisée")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Retélécharger et extraire")
    args = parser.parse_args()

    name = args.name or args.dataset
    if args.url and not args.name:
        parser.error("--name est requis avec --url")
    url = args.url or DATASET_URLS[args.dataset]
    download_dataset(name, url, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
