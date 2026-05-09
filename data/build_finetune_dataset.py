"""
Build finetuning dataset for Xyralis LFM2-VL model.
Loads samples from Sentinel-2 sources, computes spectral indices, generates prompts.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger("xyralis.build_dataset")

# ── Stress classification rules ────────────────────────────────────────────────
def label_to_stress_class(raw_label: str, indices: Dict[str, float]) -> Tuple[str, int, str]:
    """
    Classify crop stress level from BOTH raw label AND spectral indices.
    Returns (stress_class, severity_0_to_2, reasoning)
    """
    ndvi = indices.get("ndvi_mean", 0.0)
    ndwi = indices.get("ndwi_mean", 0.0)
    stress_score = indices.get("stress_score", 0.0)

    # Incorporate raw_label hints (e.g., BigEarthNet labels give crop type context)
    label_lower = raw_label.lower()

    # Base classification on indices
    if ndvi > 0.5 and ndwi > -0.1:
        cls = "healthy"
        severity = 0
        base_reason = f"NDVI={ndvi:.3f} (vigorous), NDWI={ndwi:.3f} (no water stress)"
    elif (0.3 <= ndvi <= 0.5) or (ndvi > 0.5 and ndwi <= -0.2):
        cls = "mild_stress"
        severity = 1
        stress_type = "water stress" if ndwi <= -0.2 else "moderate vigor loss"
        base_reason = f"NDVI={ndvi:.3f} (moderate), {stress_type} (NDWI={ndwi:.3f})"
    elif ndvi < 0.3 or stress_score > 70:
        cls = "severe_stress"
        severity = 2
        cause = "low vigor" if ndvi < 0.3 else "high stress score"
        base_reason = f"NDVI={ndvi:.3f} ({cause}), combined stress_score={stress_score:.1f}"
    else:
        cls = "mild_stress"
        severity = 1
        base_reason = f"Unclear case — assigned mild_stress (NDVI={ndvi:.3f}, NDWI={ndwi:.3f})"

    # Adjust based on raw_label if it provides strong contradictory evidence
    # Ejemplo: si la etiqueta dice "Permanent crops" pero NDVI muy bajo, puede indicar estrés severo
    if "permanent" in label_lower or "arable" in label_lower:
        if ndvi < 0.4:
            # Cultivos de sembra a veces tienen NDVI más bajo en ciertas épocas
            reasoning = f"{base_reason} [label hint: {raw_label} may have seasonal low canopy]"
        else:
            reasoning = f"{base_reason} [label: {raw_label}]"
    else:
        reasoning = f"{base_reason} [label: {raw_label}]"

    return cls, severity, reasoning


# ── Prompt generation ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert agronomist specializing in remote sensing and precision agriculture. "
    "You analyze multispectral satellite imagery to assess crop health and provide actionable recommendations. "
    "Always respond with a structured JSON containing exactly these keys: "
    "\"classification\" (one of: healthy, mild_stress, severe_stress), "
    "\"confidence\" (integer percentage 0-100), "
    "\"recommended_action\" (concise advice, max 100 chars), "
    "\"days_to_action\" (integer days)."
)

NEGATIVE_EXAMPLES = {
    "healthy": [
        {
            "input": "NDVI=0.85, NDWI=0.05 — very high vigor, adequate water",
            "wrong_output": {"classification": "mild_stress", "confidence": 60},
            "explanation": "High NDVI >0.8 indicates healthy canopy, not stressed"
        },
        {
            "input": "NDVI=0.72, EVI=0.65 — dense green vegetation",
            "wrong_output": {"classification": "severe_stress", "confidence": 90},
            "explanation": "Both NDVI and EVI are well above thresholds for healthy crops"
        }
    ],
    "mild_stress": [
        {
            "input": "NDVI=0.45, NDWI=-0.15 — moderate vigor, slight water deficit",
            "wrong_output": {"classification": "healthy", "confidence": 85},
            "explanation": "NDVI 0.45 is below healthy threshold of 0.5, mild water stress present"
        },
        {
            "input": "NDVI=0.55, NDWI=-0.25 — good vigor but water stress starting",
            "wrong_output": {"classification": "severe_stress", "confidence": 80},
            "explanation": "NDVI >0.5 but NDWI < -0.2 indicates early water deficit, not severe yet"
        }
    ],
    "severe_stress": [
        {
            "input": "NDVI=0.22, stress_score=82 — very low vigor, high stress",
            "wrong_output": {"classification": "mild_stress", "confidence": 70},
            "explanation": "NDVI <0.3 and stress_score >70 both indicate severe condition"
        },
        {
            "input": "NDVI=0.18, NDWI=-0.35 — wilting, severe drought",
            "wrong_output": {"classification": "healthy", "confidence": 50},
            "explanation": "Extremely low NDVI combined with very negative NDWI cannot be healthy"
        }
    ],
}


def build_prompt(indices: Dict[str, float], stress_class: str, reasoning: str) -> Tuple[str, str, str]:
    """
    Generate (system_prompt, user_prompt, expected_response) for one sample.
    """
    rng = np.random.RandomState(int(indices.get("ndvi_mean", 0) * 10000) % (2**32))

    idx_text = (
        f"NDVI={indices.get('ndvi_mean', 0):.3f} (σ={indices.get('ndvi_std', 0):.3f}), "
        f"NDWI={indices.get('ndwi_mean', 0):.3f}, "
        f"SWIR={indices.get('swir_ratio_mean', 0):.3f}, "
        f"EVI={indices.get('evi_mean', 0):.3f}, "
        f"Stress={indices.get('stress_score', 0):.1f}/100"
    )

    templates = [
        f"Spectral indices: {idx_text}. Classify crop stress status.",
        f"Given these vegetation indices: {idx_text}. Assess health and recommend intervention.",
        f"Crop condition analysis: {idx_text}. Determine stress level and timing for action.",
    ]
    user_prompt = rng.choice(templates)

    expected_response = (
        f"Classification: {stress_class}\n"
        f"Confidence: {rng.randint(75, 99)}%\n"
        f"Recommended action: {reasoning}\n"
        f"days_to_action: {rng.choice([1, 3, 5, 7, 10])}"
    )

    return SYSTEM_PROMPT, user_prompt, expected_response


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all_samples(base_dir: str = "data/raw") -> List[Dict]:
    """
    Load samples from:
      - BigEarthNet: .npy + .json sidecars
      - SimSat demo: manifest.json
    Returns list of dicts with image_path, indices, raw_label, source, split (to be assigned).
    """
    samples: List[Dict] = []
    base = Path(base_dir)

    # 1. BigEarthNet
    bigearth_dir = base / "bigearth"
    if bigearth_dir.exists():
        for npy_file in bigearth_dir.glob("*.npy"):
            json_file = npy_file.with_suffix(".json")
            if json_file.exists():
                with open(json_file) as f:
                    meta = json.load(f)
                sample = {
                    "image_path": str(npy_file),
                    "raw_label": ", ".join(meta.get("labels", [])),
                    "split": None,  # to be assigned
                    "source": "bigearth",
                    "indices": None,  # compute later
                }
                samples.append(sample)

    # 2. SimSat demo
    simsat_manifest = base / "simsat_demo" / "manifest.json"
    if simsat_manifest.exists():
        with open(simsat_manifest) as f:
            manifest = json.load(f)
        by_parcel: Dict[str, Dict] = {}
        for entry in manifest:
            if entry["type"] == "false_color":
                name = entry["name"]
                by_parcel.setdefault(name, {})["image"] = entry["path"]
            elif entry["type"] == "indices":
                name = entry["name"]
                by_parcel.setdefault(name, {})["indices"] = entry["path"]
            elif entry["type"] == "bands":
                name = entry["name"]
                by_parcel.setdefault(name, {})["bands"] = entry["path"]

        for name, data in by_parcel.items():
            if "indices" in data:
                with open(data["indices"]) as f:
                    indices = json.load(f)
                sample = {
                    "image_path": data.get("image", ""),
                    "raw_label": f"simsat_{name}",
                    "split": None,
                    "source": "simsat",
                    "indices": indices,
                }
                samples.append(sample)

    logger.info("Loaded %d samples from all sources (no split assigned yet)", len(samples))
    return samples


# ── Spectral indices compute (fallback) ────────────────────────────────────────
def compute_indices_from_array(image_path: str) -> Optional[Dict[str, float]]:
    """
    Load .npy array and compute spectral indices.
    Supports BigEarthNet (13 bands) and SimSat (6 bands).
    """
    try:
        arr = np.load(image_path)
        # Normalize if needed (Sentinel-2 L2A values typically 0-10000)
        if arr.max() > 1.1:
            arr = arr.astype(np.float32) / 10000.0

        # Select bands: red, green, blue, nir, swir16, swir22
        if arr.shape[0] >= 12:  # BigEarthNet: B01..B12, B8A
            # Band mapping for Sentinel-2 L2A:
            # B02=blue, B03=green, B04=red, B08=nir, B11=swir16, B12=swir22
            band_indices = [3, 2, 1, 7, 11, 12]  # 0-based indices
            bands = np.stack([arr[i] for i in band_indices], axis=0)
        else:
            bands = arr  # Assume already [6,H,W]

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from api.simsat_client import compute_spectral_indices

        return compute_spectral_indices(bands)

    except Exception as e:
        logger.warning("Failed to compute indices for %s: %s", image_path, e)
        return None


# ── Dataset building ───────────────────────────────────────────────────────────
def assign_splits(samples: List[Dict], seed: int = 42) -> None:
    """
    Assign train/val/test splits (70/15/15) to samples that have split=None.
    Modifies samples in-place.
    """
    unassigned = [i for i, s in enumerate(samples) if s.get("split") is None]
    n = len(unassigned)
    if n == 0:
        return

    np.random.seed(seed)
    np.random.shuffle(unassigned)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    for idx_pos, sample_idx in enumerate(unassigned):
        if idx_pos < n_train:
            samples[sample_idx]["split"] = "train"
        elif idx_pos < n_train + n_val:
            samples[sample_idx]["split"] = "val"
        else:
            samples[sample_idx]["split"] = "test"

    logger.info("Assigned splits: train=%d, val=%d, test=%d", n_train, n_val, n - n_train - n_val)


def save_dataset(
    samples: List[Dict],
    output_dir: str = "data/dataset",
    variants_per_sample: int = 3
) -> Dict[str, Any]:
    """
    Save train.jsonl, val.jsonl, test.jsonl and dataset_stats.json.
    Each line: {"image": path, "system": system_prompt, "prompt": user_prompt,
               "response": expected_response, "label": stress_class, "indices": {...}}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    neg_examples = NEGATIVE_EXAMPLES.copy()
    train_lines, val_lines, test_lines = [], [], []
    stats = {
        "total": 0,
        "by_class": {"healthy": 0, "mild_stress": 0, "severe_stress": 0},
        "by_split": {"train": 0, "val": 0, "test": 0},
        "index_stats": {},
    }

    for sample in samples:
        # Compute indices if missing
        if sample.get("indices") is None:
            sample["indices"] = compute_indices_from_array(sample["image_path"])
            if sample["indices"] is None:
                logger.warning("Skipping sample (indices compute failed): %s", sample["image_path"])
                continue

        # Classify using BOTH raw_label AND indices
        stress_class, severity, reasoning = label_to_stress_class(sample["raw_label"], sample["indices"])

        # Generate prompt variants
        variants = [build_prompt(sample["indices"], stress_class, reasoning) for _ in range(variants_per_sample)]

        split = sample.get("split", "train")  # after assign_splits, all should have split

        for sys_prompt, user_prompt, expected in variants:
            line = {
                "image": sample["image_path"],
                "system": sys_prompt,
                "prompt": user_prompt,
                "response": expected,
                "label": stress_class,
                "severity": severity,
                "indices": sample["indices"],
            }

            # Add negative examples only to train
            if split == "train" and stress_class in neg_examples and neg_examples[stress_class]:
                for neg in neg_examples[stress_class]:
                    neg_line = {
                        "image": "",
                        "system": sys_prompt,
                        "prompt": f"Indices: {neg['input']}. Classify.",
                        "response": f"Classification: {stress_class}\nConfidence: 95%\nRecommended action: {neg['explanation']}\ndays_to_action: 5",
                        "label": stress_class,
                        "severity": severity,
                        "indices": sample["indices"].copy(),
                        "is_negative_example": True,
                    }
                    train_lines.append(neg_line)
                neg_examples[stress_class] = []  # only add once per class total

            if split == "train":
                train_lines.append(line)
            elif split == "val":
                val_lines.append(line)
            else:
                test_lines.append(line)

        stats["by_class"][stress_class] += variants_per_sample
        stats["by_split"][split] += variants_per_sample
        stats["total"] += variants_per_sample

    # Write files
    for split_name, lines in [("train", train_lines), ("val", val_lines), ("test", test_lines)]:
        if not lines:
            continue
        path = out / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        logger.info("Wrote %s.jsonl: %d samples", split_name, len(lines))

    # Per-class index statistics
    class_indices: Dict[str, List[Dict]] = {c: [] for c in ["healthy", "mild_stress", "severe_stress"]}
    for line in train_lines + val_lines + test_lines:
        cls = line["label"]
        idx = line.get("indices", {})
        if idx:
            class_indices[cls].append(idx)

    stats["index_stats"] = {}
    for cls, idx_list in class_indices.items():
        if idx_list:
            ndvi_vals = [i["ndvi_mean"] for i in idx_list]
            stats["index_stats"][cls] = {
                "n": len(idx_list),
                "ndvi_mean": float(np.mean(ndvi_vals)),
                "ndvi_std": float(np.std(ndvi_vals)),
            }

    stats_path = out / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset built: %s", json.dumps(stats, indent=2))
    return stats


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build finetuning dataset for Xyralis")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/dataset", help="Output dataset directory")
    parser.add_argument("--variants", type=int, default=3, help="Number of prompt variants per sample")
    args = parser.parse_args()

    logger.info("=== Building Xyralis Finetune Dataset ===")
    logger.info("Raw dir: %s", args.raw_dir)
    logger.info("Output : %s", args.output_dir)

    samples = load_all_samples(args.raw_dir)
    if not samples:
        logger.error("No samples loaded. Check raw data directories.")
        sys.exit(1)

    # Assign splits 70/15/15
    assign_splits(samples, seed=42)

    stats = save_dataset(samples, args.output_dir, args.variants)

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples (with variants): {stats['total']}")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {stats['by_split'].get(split, 0)} examples")
    print("\nClass distribution:")
    for cls, count in stats["by_class"].items():
        print(f"  {cls}: {count}")
    print("\nPer-class NDVI mean ± std:")
    for cls, ist in stats.get("index_stats", {}).items():
        print(f"  {cls}: NDVI={ist['ndvi_mean']:.3f}±{ist['ndvi_std']:.3f} (n={ist['n']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
