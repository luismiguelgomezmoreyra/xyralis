"""
Dataset integrity validation for Xyralis.
Checks: image existence, class balance, cross-split leakage, response format, NDVI ordering.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np

logger = logging.getLogger("xyralis.validate")


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        logger.error("File not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_image_exists(sample: Dict, dataset_dir: Path) -> bool:
    """Check that image_path exists relative to dataset_dir or absolutely."""
    img_path_str = sample.get("image", "")
    if not img_path_str:
        return True  # negative examples may omit image

    img_path = Path(img_path_str)
    if img_path.is_absolute():
        return img_path.exists()
    else:
        # Relative path: check against dataset_dir.parent (since images are in data/raw/)
        # Actually image_path from samples is absolute or relative to cwd
        return img_path.exists() or (dataset_dir.parent / img_path).exists()


def check_response_format(response: str) -> bool:
    return ("Classification:" in response and
            "%" in response and
            "days_to_action:" in response)


def compute_class_ndvi_means(samples: List[Dict]) -> Dict[str, float]:
    by_class: Dict[str, List[float]] = {}
    for s in samples:
        cls = s.get("label", "unknown")
        idx = s.get("indices", {})
        ndvi = idx.get("ndvi_mean")
        if ndvi is not None:
            by_class.setdefault(cls, []).append(ndvi)
    return {cls: float(np.mean(vals)) if vals else 0.0 for cls, vals in by_class.items()}


# ── Main validation ────────────────────────────────────────────────────────────
def validate_dataset_integrity(dataset_dir: str = "data/dataset") -> bool:
    base = Path(dataset_dir)
    splits = ["train", "val", "test"]
    all_samples: List[Dict] = []
    split_samples: Dict[str, List[Dict]] = {}
    errors: List[str] = []

    logger.info("=== Xyralis Dataset Integrity Validation ===")
    logger.info("Dataset dir: %s", dataset_dir)

    # Load splits
    for split in splits:
        path = base / f"{split}.jsonl"
        samples = load_jsonl(path)
        split_samples[split] = samples
        all_samples.extend(samples)
        logger.info("Loaded %s: %d samples", split, len(samples))

    if not all_samples:
        errors.append("No samples found in any split — dataset empty?")
        logger.error("No samples found")
    else:
        logger.info("Total samples: %d", len(all_samples))

    # Check 1: Image paths exist
    logger.info("\n[1/5] Checking image paths …")
    missing = []
    for sample in all_samples:
        img = sample.get("image", "")
        if img and not check_image_exists(sample, base):
            missing.append(img)
    if missing:
        errors.append(f"Missing {len(missing)} image files")
        logger.error("Missing images (first 10): %s", missing[:10])
    else:
        logger.info("  ✓ All %d image paths exist", len(all_samples))

    # Check 2: Class balance 20–60%
    logger.info("\n[2/5] Checking class balance …")
    label_counts = Counter(s["label"] for s in all_samples)
    total = sum(label_counts.values())
    for cls, cnt in label_counts.items():
        pct = 100 * cnt / total
        status = "✓" if 20 <= pct <= 60 else "✗"
        logger.info("  %s %s: %d (%.1f%%)", status, cls, cnt, pct)
        if pct < 20 or pct > 60:
            errors.append(f"Class '{cls}' imbalance: {pct:.1f}% (expected 20–60%)")

    # Check 3: No duplicates across splits
    logger.info("\n[3/5] Checking for cross-split duplicates …")
    seen: Dict[str, str] = {}
    duplicates: Set[str] = set()
    for split, samples in split_samples.items():
        for s in samples:
            img = s.get("image", "")
            if img:
                if img in seen:
                    duplicates.add(img)
                    logger.error("  Duplicate in '%s' and '%s': %s", seen[img], split, img)
                else:
                    seen[img] = split
    if duplicates:
        errors.append(f"Found {len(duplicates)} duplicate images across splits")
    else:
        logger.info("  ✓ No duplicates across splits")

    # Check 4: Response format
    logger.info("\n[4/5] Checking response format …")
    bad_format = []
    for sample in all_samples:
        resp = sample.get("response", "")
        if not check_response_format(resp):
            bad_format.append(sample.get("image", "unknown"))
    if bad_format:
        errors.append(f"{len(bad_format)} responses missing required fields")
        logger.error("Bad format samples (first 10): %s", bad_format[:10])
    else:
        logger.info("  ✓ All %d responses well-formed", len(all_samples))

    # Check 5: NDVI ordering healthy > mild > severe
    logger.info("\n[5/5] Checking NDVI mean ordering …")
    ndvi_means = compute_class_ndvi_means(all_samples)
    logger.info("  NDVI by class: %s", {k: f"{v:.3f}" for k, v in ndvi_means.items()})
    healthy = ndvi_means.get("healthy", 0)
    mild = ndvi_means.get("mild_stress", 0)
    severe = ndvi_means.get("severe_stress", 0)

    if not (healthy > mild > severe):
        errors.append(f"NDVI ordering violated: healthy={healthy:.3f}, mild={mild:.3f}, severe={severe:.3f}")
        logger.error("  ✗ Expected: healthy > mild > severe")
    else:
        logger.info("  ✓ NDVI ordering correct: %.3f > %.3f > %.3f", healthy, mild, severe)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total samples   : %d", len(all_samples))
    logger.info("Class counts    : %s", dict(label_counts))
    logger.info("Splits          : train=%d, val=%d, test=%d",
                len(split_samples["train"]), len(split_samples["val"]), len(split_samples["test"]))
    logger.info("Errors found    : %d", len(errors))
    if errors:
        logger.error("ERRORS:")
        for err in errors:
            logger.error("  ✗ %s", err)
        raise ValueError(f"Dataset validation failed with {len(errors)} errors")
    else:
        logger.info("✓ ALL CHECKS PASSED")
    logger.info("=" * 60)

    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate Xyralis dataset integrity")
    parser.add_argument("--dataset-dir", default="data/dataset", help="Dataset directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(Path("logs") / "data_pipeline.log", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        validate_dataset_integrity(args.dataset_dir)
        print("\n✓ Dataset validation PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Dataset validation FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
