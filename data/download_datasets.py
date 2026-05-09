"""
Dataset download and organization for CropAlert.
Downloads and structures training data from multiple sources: EuroSAT, BigEarthNet, SimSat demo.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_pipeline.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cropalert.download")


# ── EuroSAT ────────────────────────────────────────────────────────────────────
def download_eurosat(output_dir: str = "data/raw/eurosat") -> int:
    """
    Download EuroSAT multispectral dataset via torchvision.
    Filter to agricultural classes only: AnnualCrop, PermanentCrop, HerbaceousVegetation, Pasture.
    Balance: max 400 images per class.
    Save metadata CSV: output_dir/metadata.csv with columns: path, class, split (70/15/15).
    """
    try:
        from torchvision.datasets import EuroSAT
    except ImportError:
        logger.error("torchvision not installed. Run: pip install torchvision")
        return 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    AGRICULTURAL_CLASSES = {"AnnualCrop", "PermanentCrop", "HerbaceousVegetation", "Pasture"}
    MAX_PER_CLASS = 400

    logger.info("Downloading EuroSAT dataset …")
    dataset = EuroSAT(root=output_dir, download=True, transform=None)

    # Collect per-class samples
    samples_by_class: Dict[str, List[Dict]] = {cls: [] for cls in AGRICULTURAL_CLASSES}

    for idx, (img, label_idx) in enumerate(dataset):
        class_name = dataset.classes[label_idx]
        if class_name in AGRICULTURAL_CLASSES:
            # Determine filename
            filename = f"{class_name}_{idx:05d}.npy"
            # Save as numpy (PIL → array)
            img_array = np.array(img)  # shape [H,W,C] or [C,H,W] depending on EuroSAT version
            np.save(output_path / filename, img_array)

            samples_by_class[class_name].append({
                "path": str(output_path / filename),
                "class": class_name,
            })

    # Balance: truncate to MAX_PER_CLASS
    balanced: List[Dict] = []
    for cls, samples in samples_by_class.items():
        truncated = samples[:MAX_PER_CLASS]
        logger.info("  Class %s: %d samples (truncated to %d)", cls, len(samples), len(truncated))
        balanced.extend(truncated)

    # Shuffle and split 70/15/15
    np.random.seed(42)
    np.random.shuffle(balanced)

    n_total = len(balanced)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    for i, sample in enumerate(balanced):
        if i < n_train:
            sample["split"] = "train"
        elif i < n_train + n_val:
            sample["split"] = "val"
        else:
            sample["split"] = "test"

    # Save metadata CSV
    df = pd.DataFrame(balanced)
    csv_path = output_path / "metadata.csv"
    df.to_csv(csv_path, index=False)

    logger.info("Downloaded %d images across %d classes (CSV: %s)", n_total, len(samples_by_class), csv_path)
    return n_total


# ── BigEarthNet sample ─────────────────────────────────────────────────────────
def download_bigearth_sample(output_dir: str = "data/raw/bigearth", n_samples: int = 300) -> int:
    """
    Load BigEarthNet via streaming (torchgeo) and download first n_samples
    with agricultural labels: Arable land, Pastures, Permanent crops.
    Save each sample as .npy (12 bands) + .json sidecar.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    AGRIC_LABELS = {"Arable land", "Pastures", "Permanent crops"}

    logger.info("Loading BigEarthNet (streaming)…")
    ds = load_dataset("torchgeo/BigEarthNet", split="train", streaming=True)

    count = 0
    saved: List[Dict] = []

    logger.info("Downloading up to %d agricultural samples …", n_samples)
    for idx, example in enumerate(ds):
        if count >= n_samples:
            break

        # Check labels
        labels = example.get("labels", [])
        if not any(lbl in AGRIC_LABELS for lbl in labels):
            continue  # skip non-agricultural

        # Extract bands (12 bands as separate arrays in example)
        # BigEarthNet provides: B01, B02, ..., B12, B8A
        band_keys = [f"B{i:02d}" for i in range(1, 13)] + ["B8A"]
        band_arrays = []
        for key in band_keys:
            if key in example:
                arr = np.array(example[key])
                band_arrays.append(arr)
            else:
                logger.warning("Band %s missing in sample %d", key, idx)

        if len(band_arrays) != 13:
            logger.warning("Skipping sample %d: incomplete bands", idx)
            continue

        bands_stack = np.stack(band_arrays, axis=0).astype(np.float32)  # [13, H, W]

        # Save .npy
        sample_id = example.get("patch_id", f"bigearth_{idx:06d}")
        npy_path = output_path / f"{sample_id}.npy"
        np.save(npy_path, bands_stack)

        # Save JSON sidecar
        meta = {
            "patch_id": sample_id,
            "labels": labels,
            "source": "BigEarthNet",
            "timestamp_iso": "2026-01-01T00:00:00Z",  # dummy for consistency
        }
        json_path = output_path / f"{sample_id}.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        saved.append({"path": str(npy_path), "json": str(json_path), "labels": labels})
        count += 1

        if count % 50 == 0:
            logger.info("  Downloaded %d/%d …", count, n_samples)

    logger.info("BigEarthNet: saved %d samples to %s", count, output_dir)
    return count


# ── SimSat demo parcels ───────────────────────────────────────────────────────
DEMO_PARCELS = [
    # Valle del Mantaro, Peru
    {"lat": -12.05, "lon": -75.20, "timestamp": "2026-02-01T10:00:00Z", "name": "mantaro_pe"},
    # Corn Belt, Iowa USA
    {"lat": 42.00, "lon": -93.00, "timestamp": "2026-02-15T14:00:00Z", "name": "iowa_us"},
    # Po Valley, Italy
    {"lat": 45.10, "lon": 10.80, "timestamp": "2026-02-20T11:00:00Z", "name": "po_valley_it"},
    # Nile Delta, Egypt
    {"lat": 30.80, "lon": 31.20, "timestamp": "2026-01-10T09:00:00Z", "name": "nile_delta_eg"},
    # Mato Grosso, Brazil
    {"lat": -13.00, "lon": -55.00, "timestamp": "2026-01-25T13:00:00Z", "name": "mato_grosso_br"},
    # Punjab, India
    {"lat": 30.80, "lon": 75.50, "timestamp": "2026-03-15T10:00:00Z", "name": "punjab_in"},
    # Canterbury Plains, New Zealand
    {"lat": -43.80, "lon": 171.50, "timestamp": "2026-02-10T12:00:00Z", "name": "canterbury_nz"},
    # Andalusia, Spain
    {"lat": 37.20, "lon": -4.00, "timestamp": "2026-03-01T11:00:00Z", "name": "andalucia_es"},
    # Mekong Delta, Vietnam
    {"lat": 10.00, "lon": 105.50, "timestamp": "2026-01-20T07:00:00Z", "name": "mekong_vn"},
    # Great Plains, Kansas USA
    {"lat": 38.50, "lon": -98.00, "timestamp": "2026-06-20T14:00:00Z", "name": "kansas_us"},
    # Pampas, Argentina
    {"lat": -35.00, "lon": -64.00, "timestamp": "2026-01-05T13:00:00Z", "name": "pampas_ar"},
    # Wheatbelt, Western Australia
    {"lat": -31.50, "lon": 117.00, "timestamp": "2026-09-10T08:00:00Z", "name": "wheatbelt_au"},
    # Central Valley, California USA
    {"lat": 36.50, "lon": -119.50, "timestamp": "2026-04-15T14:00:00Z", "name": "central_valley_us"},
    # Campine, Belgium
    {"lat": 51.20, "lon": 4.80, "timestamp": "2026-05-20T10:00:00Z", "name": "campine_be"},
    # Black Sea Region, Turkey
    {"lat": 41.50, "lon": 36.00, "timestamp": "2026-06-10T11:00:00Z", "name": "blacksea_tr"},
]


def download_simsat_demo_parcels(output_dir: str = "data/raw/simsat_demo") -> int:
    """
    Fetch Sentinel-2 imagery for 15 predefined agricultural parcels via SimSatClient.
    Saves for each parcel:
      - false_color PNG
      - raw bands .npy (6 bands: R,G,B,NIR,SWIR16,SWIR22)
      - indices .json
    Creates manifest.json in output_dir.
    """
    try:
        # Import SimSatClient from project
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from api.simsat_client import SimSatClient
    except ImportError as e:
        logger.error("Failed to import SimSatClient: %s", e)
        return 0

    client = SimSatClient()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    saved = 0

    logger.info("Fetching SimSat demo parcels (%d total)…", len(DEMO_PARCELS))

    for parcel in DEMO_PARCELS:
        lat = parcel["lat"]
        lon = parcel["lon"]
        ts = parcel["timestamp"]
        name = parcel["name"]

        logger.info("  Fetching %s (%.4f, %.4f)…", name, lat, lon)

        try:
            result = client.fetch_sentinel_at(lat=lat, lon=lon, timestamp_iso=ts, size_km=5.0)

            if not result.image_available:
                logger.warning("  %s: image not available (cloud_cover=%s%%) — skipping", name, result.cloud_cover)
                continue

            # Save raw bands array
            if result.bands_array is not None:
                npy_path = output_path / f"{name}_bands.npy"
                np.save(npy_path, result.bands_array)
                manifest.append({"name": name, "type": "bands", "path": str(npy_path)})
                saved += 1

            # Save false-color PNG
            if result.false_color_png_path and Path(result.false_color_png_path).exists():
                png_path = output_path / f"{name}_falsecolor.png"
                # Move from temp location to output
                import shutil
                shutil.copy(result.false_color_png_path, png_path)
                manifest.append({"name": name, "type": "false_color", "path": str(png_path)})

            # Save indices JSON
            if result.indices:
                idx_path = output_path / f"{name}_indices.json"
                with open(idx_path, "w") as f:
                    json.dump(result.indices, f, indent=2)
                manifest.append({"name": name, "type": "indices", "path": str(idx_path)})

            logger.info("  ✓ %s saved", name)

        except Exception as e:
            logger.error("  ✗ %s failed: %s", name, e)

    # Write manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("SimSat demo: saved %d parcels to %s", saved, output_dir)
    return saved


# ── Main CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download datasets for CropAlert")
    parser.add_argument("--source", choices=["eurosat", "bigearth", "simsat", "all"], default="all",
                        help="Dataset source to download")
    parser.add_argument("--output-dir", default="data/raw", help="Base output directory")
    args = parser.parse_args()

    logger.info("=== CropAlert Dataset Downloader ===")
    logger.info("Source: %s", args.source)
    logger.info("Output : %s", args.output_dir)

    total = 0

    if args.source in ("eurosat", "all"):
        logger.info("\n--- EuroSAT ---")
        n = download_eurosat(os.path.join(args.output_dir, "eurosat"))
        total += n

    if args.source in ("bigearth", "all"):
        logger.info("\n--- BigEarthNet ---")
        n = download_bigearth_sample(os.path.join(args.output_dir, "bigearth"), n_samples=300)
        total += n

    if args.source in ("simsat", "all"):
        logger.info("\n--- SimSat Demo ---")
        n = download_simsat_demo_parcels(os.path.join(args.output_dir, "simsat_demo"))
        total += n

    logger.info("\n=== Total samples downloaded: %d ===", total)


if __name__ == "__main__":
    main()
