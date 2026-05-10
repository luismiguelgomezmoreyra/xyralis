"""
Xyralis Dataset Validator.
Checks integrity and consistency of the generated fine-tuning dataset.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

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
logger = logging.getLogger("xyralis.validate")

def validate_dataset_integrity(dataset_dir: str = "data/dataset"):
    """
    Perform multiple integrity checks on the dataset.
    """
    base = Path(dataset_dir)
    splits = ["train", "val", "test"]
    all_lines = []
    split_data = {}
    
    logger.info("=== Starting Dataset Validation ===")
    
    # Load and basic check
    for split in splits:
        path = base / f"{split}.jsonl"
        if not path.exists():
            logger.error("Missing split file: %s", path)
            raise ValueError(f"Missing split: {split}")
            
        lines = []
        with open(path) as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    lines.append(data)
                except Exception as e:
                    logger.error("JSON error in %s line %d: %s", split, i, e)
                    raise ValueError(f"Invalid JSON in {split}")
        
        split_data[split] = lines
        all_lines.extend(lines)
        logger.info("  %s: %d lines", split, len(lines))

    # 1. Image Path Existence
    logger.info("Checking image path existence…")
    missing_images = 0
    for line in all_lines:
        img = line.get("image", "")
        if img: # skip negative examples with empty image
            if not Path(img).exists():
                missing_images += 1
                if missing_images <= 5:
                    logger.warning("    Missing image: %s", img)
    
    if missing_images > 0:
        logger.error("Found %d missing image files", missing_images)
        raise ValueError(f"CRITICAL: {missing_images} image files missing on disk.")
    else:
        logger.info("  ✓ All image paths exist.")

    # 2. Class Balance (20% - 60%)
    logger.info("Checking class balance…")
    labels = [line["label"] for line in all_lines]
    unique_labels = set(labels)
    total = len(labels)
    
    for lbl in unique_labels:
        count = labels.count(lbl)
        pct = (count / total) * 100
        logger.info("    %s: %d (%.1f%%)", lbl, count, pct)
        if pct < 20 or pct > 60:
             logger.error("    Class '%s' is imbalanced: %.1f%%", lbl, pct)
             raise ValueError(f"CRITICAL: Class imbalance: {lbl} ({pct:.1f}%)")
    
    # 3. No Leakage
    logger.info("Checking for cross-split duplicates…")
    train_imgs = set(l["image"] for l in split_data["train"] if l["image"])
    val_imgs = set(l["image"] for l in split_data["val"] if l["image"])
    test_imgs = set(l["image"] for l in split_data["test"] if l["image"])
    
    leak_train_val = train_imgs.intersection(val_imgs)
    leak_train_test = train_imgs.intersection(test_imgs)
    leak_val_test = val_imgs.intersection(test_imgs)
    
    if leak_train_val or leak_train_test or leak_val_test:
        logger.error("Found image leakage between splits!")
        if leak_train_val: logger.error("    Train/Val leakage: %d images", len(leak_train_val))
        raise ValueError("Data leakage detected")
    else:
        logger.info("  ✓ No cross-split leakage.")

    # 4. Response Format
    logger.info("Checking response formats…")
    bad_formats = 0
    for line in all_lines:
        resp = line["response"]
        checks = [
            "Classification:" in resp,
            "%" in resp,
            "days_to_action:" in resp
        ]
        if not all(checks):
            bad_formats += 1
            
    if bad_formats > 0:
        logger.error("Found %d responses with invalid format", bad_formats)
        raise ValueError("Response format error")
    else:
        logger.info("  ✓ All responses well-formatted.")

    # 5. NDVI Ordering
    logger.info("Checking NDVI ordering consistency…")
    ndvi_by_class = {"healthy": [], "mild_stress": [], "severe_stress": []}
    for line in all_lines:
        lbl = line["label"]
        if lbl in ndvi_by_class:
            ndvi_by_class[lbl].append(line["indices"].get("ndvi_mean", 0))
            
    means = {k: np.mean(v) if v else 0 for k, v in ndvi_by_class.items()}
    logger.info("    Means: healthy=%.3f, mild=%.3f, severe=%.3f", 
                means["healthy"], means["mild_stress"], means["severe_stress"])
                
    if not (means["healthy"] > means["mild_stress"] > means["severe_stress"]):
        logger.error("NDVI order violation: Expected healthy > mild > severe")
        raise ValueError(f"CRITICAL: Biological NDVI ordering violated: {means}")
    else:
        logger.info("  ✓ NDVI ordering is consistent.")

    logger.info("=== ✓ ALL CRITICAL CHECKS PASSED ===")

def main():
    parser = argparse.ArgumentParser(description="Xyralis Dataset Validator")
    parser.add_argument("--dataset-dir", default="data/dataset")
    args = parser.parse_args()
    
    try:
        validate_dataset_integrity(args.dataset_dir)
    except Exception as e:
        logger.error("Validation FAILED: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
