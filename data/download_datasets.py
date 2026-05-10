"""
Xyralis Download Pipeline.
Downloads and organizes training data from EuroSAT, BigEarthNet, and SimSat.
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup Logging
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
logger = logging.getLogger("xyralis.download")

# ── EuroSAT ───────────────────────────────────────────────────────────────────
def download_eurosat(output_dir: str = "data/raw/eurosat") -> int:
    """
    Download EuroSAT multispectral dataset using torchvision.
    Filters for agricultural classes and caps at 400 per class.
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError:
        logger.error("torch/torchvision not installed. Run: pip install torch torchvision")
        return 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    TARGET_CLASSES = {"AnnualCrop", "PermanentCrop", "HerbaceousVegetation", "Pasture"}
    MAX_PER_CLASS = 400

    logger.info("Downloading EuroSAT (Multispectral)…")
    # EuroSAT in torchvision is RGB by default unless specified or using the MS version
    # Note: torchvision.datasets.EuroSAT downloads the whole dataset.
    try:
        # We use the dataset to get the file list.
        # MS version is usually a separate download, but for this pipeline 
        # we will simulate the organization logic if already downloaded or handle the download.
        ds = datasets.EuroSAT(root=output_path, download=True)
    except Exception as e:
        logger.error("EuroSAT download failed: %s", e)
        return 0

    metadata = []
    class_counts = {cls: 0 for cls in TARGET_CLASSES}
    
    # Organize and filter
    for img_path, label_idx in ds.samples:
        class_name = ds.classes[label_idx]
        if class_name not in TARGET_CLASSES:
            continue
        
        if class_counts[class_name] >= MAX_PER_CLASS:
            continue
            
        # Determine split (70/15/15)
        rnd = np.random.random()
        if rnd < 0.70:
            split = "train"
        elif rnd < 0.85:
            split = "val"
        else:
            split = "test"
            
        metadata.append({
            "path": img_path,
            "class": class_name,
            "split": split
        })
        class_counts[class_name] += 1

    # Save metadata CSV
    df = pd.DataFrame(metadata)
    csv_path = output_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    
    total = len(metadata)
    logger.info("Downloaded %d images across %d classes", total, len(class_counts))
    return total

# ── BigEarthNet ────────────────────────────────────────────────────────────────
def download_bigearth_sample(output_dir: str = "data/raw/bigearth", n_samples: int = 300) -> int:
    """
    Load BigEarthNet via streaming and download agricultural samples.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    AGRIC_LABELS = {"Arable land", "Pastures", "Permanent crops"}
    
    logger.info("Loading BigEarthNet (streaming mode, 120GB source)…")
    try:
        ds = load_dataset("torchgeo/BigEarthNet", split="train", streaming=True)
    except Exception as e:
        logger.error("BigEarthNet load failed: %s", e)
        return 0

    count = 0
    pbar = tqdm(total=n_samples, desc="BigEarthNet")
    
    it = iter(ds)
    while count < n_samples:
        try:
            example = next(it)
            labels = example.get("labels", [])
            if not any(lbl in AGRIC_LABELS for lbl in labels):
                continue
                
            # Extract 12 bands
            band_keys = [f"B{i:02d}" for i in range(1, 13)]
            band_arrays = []
            for key in band_keys:
                if key in example:
                    band_arrays.append(np.array(example[key]))
            
            if len(band_arrays) < 12:
                continue
                
            bands_stack = np.stack(band_arrays, axis=0).astype(np.float32)
            sample_id = example.get("patch_id", f"be_{count:06d}")
            
            # Save .npy
            npy_path = output_path / f"{sample_id}.npy"
            np.save(npy_path, bands_stack)
            
            # Save sidecar JSON
            meta = {
                "patch_id": sample_id,
                "labels": labels,
                "source": "BigEarthNet",
                "timestamp_iso": "2026-01-01T00:00:00Z"
            }
            with open(output_path / f"{sample_id}.json", "w") as f:
                json.dump(meta, f, indent=2)
                
            count += 1
            pbar.update(1)
            if count % 50 == 0:
                logger.info("  BigEarth: Downloaded %d/%d samples", count, n_samples)
        except StopIteration:
            break
        except Exception as e:
            logger.warning("  BigEarth: Network or processing error on sample, skipping: %s", e)
            continue
            
    pbar.close()
    logger.info("BigEarthNet: Saved %d samples", count)
    return count

# ── SimSat Demo ───────────────────────────────────────────────────────────────
DEMO_PARCELS = [
    {"lat": -12.05, "lon": -75.2, "ts": "2026-02-01T10:00:00Z", "name": "mantaro_pe"},
    {"lat": 42.0, "lon": -93.0, "ts": "2026-02-15T14:00:00Z", "name": "corn_belt_ia"},
    {"lat": 45.1, "lon": 10.8, "ts": "2026-02-20T11:00:00Z", "name": "po_valley_it"},
    {"lat": 30.8, "lon": 31.2, "ts": "2026-01-10T09:00:00Z", "name": "nile_delta_eg"},
    {"lat": -13.0, "lon": -55.0, "ts": "2026-01-25T13:00:00Z", "name": "mato_grosso_br"},
    {"lat": 30.8, "lon": 75.5, "ts": "2026-03-15T10:00:00Z", "name": "punjab_in"},
    {"lat": -43.8, "lon": 171.5, "ts": "2026-02-10T12:00:00Z", "name": "canterbury_nz"},
    {"lat": 37.2, "lon": -4.0, "ts": "2026-03-01T11:00:00Z", "name": "andalucia_es"},
    {"lat": 10.0, "lon": 105.5, "ts": "2026-01-20T07:00:00Z", "name": "mekong_vn"},
    {"lat": 38.5, "lon": -98.0, "ts": "2026-06-20T14:00:00Z", "name": "kansas_us"},
    {"lat": -35.0, "lon": -64.0, "ts": "2026-01-05T13:00:00Z", "name": "pampas_ar"},
    {"lat": -31.5, "lon": 117.0, "ts": "2026-09-10T08:00:00Z", "name": "wheatbelt_au"},
    {"lat": 36.5, "lon": -119.5, "ts": "2026-04-15T14:00:00Z", "name": "central_valley_us"},
    {"lat": 51.2, "lon": 4.8, "ts": "2026-05-20T10:00:00Z", "name": "campine_be"},
    {"lat": 41.5, "lon": 36.0, "ts": "2026-06-10T11:00:00Z", "name": "blacksea_tr"},
]

def download_simsat_demo_parcels(output_dir: str = "data/raw/simsat_demo") -> int:
    """
    Fetch Sentinel-2 imagery for 15 predefined parcels via SimSatClient.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from api.simsat_client import SimSatClient
    except ImportError as e:
        logger.error("SimSatClient import failed: %s", e)
        return 0

    client = SimSatClient()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    saved = 0

    logger.info("Fetching %d SimSat demo parcels…", len(DEMO_PARCELS))

    for p in DEMO_PARCELS:
        logger.info("  Fetching %s (%s, %s)…", p["name"], p["lat"], p["lon"])
        try:
            result = client.fetch_sentinel_at(
                lat=p["lat"], 
                lon=p["lon"], 
                timestamp_iso=p["ts"]
            )

            if not result.image_available:
                logger.warning("  %s: Image not available, skipping", p["name"])
                continue

            # Save PNG
            if result.false_color_png_path:
                import shutil
                png_dst = output_path / f"{p['name']}.png"
                shutil.copy(result.false_color_png_path, png_dst)
            
            # Save NPY
            npy_path = output_path / f"{p['name']}.npy"
            if result.bands_array is not None:
                np.save(npy_path, result.bands_array)
            
            # Save Indices
            idx_path = output_path / f"{p['name']}_indices.json"
            with open(idx_path, "w") as f:
                json.dump(result.indices, f, indent=2)

            manifest.append({
                "name": p["name"],
                "image_path": str(png_dst),
                "npy_path": str(npy_path),
                "indices_path": str(idx_path),
                "source": "simsat"
            })
            saved += 1
            logger.info("  ✓ %s saved", p["name"])

        except Exception as e:
            logger.error("  ✗ %s failed: %s", p["name"], e)

    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("SimSat Demo: Saved %d parcels", saved)
    return saved

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Xyralis Data Downloader")
    parser.add_argument("--source", choices=["eurosat", "bigearth", "simsat", "all"], default="all")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    total = 0
    if args.source in ("eurosat", "all"):
        total += download_eurosat(os.path.join(args.output_dir, "eurosat"))
    if args.source in ("bigearth", "all"):
        total += download_bigearth_sample(os.path.join(args.output_dir, "bigearth"))
    if args.source in ("simsat", "all"):
        total += download_simsat_demo_parcels(os.path.join(args.output_dir, "simsat_demo"))

    logger.info("Download pipeline complete. Total samples: %d", total)

if __name__ == "__main__":
    main()
