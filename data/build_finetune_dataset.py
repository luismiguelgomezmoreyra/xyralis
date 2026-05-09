"""
CropAlert Dataset Builder.
Processes raw data into JSONL format for LFM2-VL fine-tuning.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

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
logger = logging.getLogger("cropalert.build")

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_indices_for_npy(npy_path: str) -> Optional[Dict[str, float]]:
    """Compute spectral indices from .npy array (12 or 13 bands)."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from api.simsat_client import compute_spectral_indices
        
        arr = np.load(npy_path)
        # BigEarthNet has 12 bands. SimSat usually has 6 or 12.
        # Ensure we have the right mapping.
        # For simplicity, if it's 12 bands, we assume Sentinel-2 standard order B1-B12.
        # If it's 6 bands, we assume [R, G, B, NIR, SWIR1, SWIR2].
        return compute_spectral_indices(arr)
    except Exception as e:
        logger.warning("Failed to compute indices for %s: %s", npy_path, e)
        return None

# ── Stress Classification ───────────────────────────────────────────────────────
def label_to_stress_class(raw_label: str, indices: Dict[str, float]) -> Tuple[str, int, str]:
    """
    Classify crop stress based on label and indices.
    Returns (stress_class, severity_0_to_2, reasoning)
    """
    ndvi = indices.get("ndvi_mean", 0.0)
    ndwi = indices.get("ndwi_mean", 0.0)
    stress_score = indices.get("stress_score", 0.0)
    
    # Item 5: Use label context to adjust thresholds
    # Permanent crops/Pastures often maintain lower NDVI than dense Annual crops
    # We use the label to set a more robust baseline.
    if "AnnualCrop" in raw_label or "Arable" in raw_label:
        healthy_threshold = 0.55
        mild_threshold = 0.35
    else:
        healthy_threshold = 0.45
        mild_threshold = 0.25

    # Logic
    if ndvi > healthy_threshold and ndwi > -0.1:
        cls, sev = "healthy", 0
        reason = f"High vigor (NDVI={ndvi:.2f} > {healthy_threshold}) and adequate water content (NDWI={ndwi:.2f})."
    elif (mild_threshold <= ndvi <= healthy_threshold) or (ndvi > healthy_threshold and ndwi < -0.2):
        cls, sev = "mild_stress", 1
        if ndwi < -0.2:
            reason = f"Vigorous canopy but significant water deficit (NDWI={ndwi:.2f})."
        else:
            reason = f"Moderate vigor loss (NDVI={ndvi:.2f}) indicating early stress."
    elif ndvi < mild_threshold or stress_score > 70:
        cls, sev = "severe_stress", 2
        reason = f"Critical vigor loss (NDVI={ndvi:.2f} < {mild_threshold}) or high stress composite ({stress_score:.1f})."
    else:
        cls, sev = "mild_stress", 1
        reason = "Unclear signal — defaulting to mild_stress for safety."
        
    # Incorporate raw label context into reasoning
    reason = f"Context: {raw_label}. {reason}"
    
    return cls, sev, reason

# ── Prompt Generation ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert agronomist specializing in remote sensing. "
    "Your task is to analyze satellite imagery indices and provide a structured diagnosis. "
    "Always provide: Classification (healthy, mild_stress, severe_stress), "
    "Confidence (%), Recommended action, and days_to_action (integer)."
)

NEGATIVE_EXAMPLES = [
    {
        "indices": {"ndvi_mean": 0.05, "ndwi_mean": 0.0, "stress_score": 0.0},
        "label": "insufficient data",
        "reason": "Spectral values are consistent with non-vegetated surfaces or water."
    },
    {
        "indices": {"ndvi_mean": 0.99, "ndwi_mean": 0.8, "stress_score": 0.0},
        "label": "insufficient data",
        "reason": "Possible sensor saturation or anomaly; NDVI/NDWI values are outside normal crop ranges."
    }
]

def build_prompt(indices: Dict[str, float], stress_class: str, reasoning: str, v_idx: int = 0) -> Tuple[str, str, str]:
    """Generate (system, user, response) with 3 variants."""
    
    idx_str = ", ".join([f"{k}={v:.3f}" for k, v in indices.items()])
    
    user_templates = [
        f"Analyze these spectral indices: {idx_str}. What is the crop status?",
        f"Given the following vegetation metrics: {idx_str}. Assess health and suggest action.",
        f"Based on the remote sensing data ({idx_str}), provide a detailed agronomical diagnosis."
    ]
    
    user_prompt = user_templates[v_idx % 3]
    
    conf = np.random.randint(75, 99) if stress_class != "insufficient data" else 95
    days = np.random.choice([1, 3, 7, 10]) if stress_class != "insufficient data" else 0
    
    expected_response = (
        f"Classification: {stress_class}\n"
        f"Confidence: {conf}%\n"
        f"Recommended action: {reasoning}\n"
        f"days_to_action: {days}"
    )
    
    return SYSTEM_PROMPT, user_prompt, expected_response

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_all_samples(raw_dir: str = "data/raw") -> List[Dict]:
    """Load from EuroSAT, BigEarth, and SimSat."""
    samples = []
    base = Path(raw_dir)
    
    # 1. EuroSAT
    eurosat_csv = base / "eurosat" / "metadata.csv"
    if eurosat_csv.exists():
        df = pd.read_csv(eurosat_csv)
        for _, row in df.iterrows():
            # For EuroSAT (RGB/PNG), we might need indices from elsewhere or compute proxy
            # In a real pipeline, we'd use the .npy version if available.
            # Here we try to find a .npy sibling if it exists.
            npy_path = Path(row["path"]).with_suffix(".npy")
            indices = compute_indices_for_npy(str(npy_path)) if npy_path.exists() else {"ndvi_mean": 0.6, "ndwi_mean": 0.0, "stress_score": 10.0}
            
            samples.append({
                "image_path": row["path"],
                "indices": indices,
                "raw_label": row["class"],
                "split": row["split"]
            })
            
    # 2. BigEarthNet
    be_dir = base / "bigearth"
    if be_dir.exists():
        for json_file in be_dir.glob("*.json"):
            with open(json_file) as f:
                meta = json.load(f)
            npy_path = json_file.with_suffix(".npy")
            indices = compute_indices_for_npy(str(npy_path))
            if indices:
                samples.append({
                    "image_path": str(npy_path),
                    "indices": indices,
                    "raw_label": ", ".join(meta["labels"]),
                    "split": "train" # BigEarth is only train in this demo
                })
                
    # 3. SimSat
    simsat_manifest = base / "simsat_demo" / "manifest.json"
    if simsat_manifest.exists():
        with open(simsat_manifest) as f:
            manifest = json.load(f)
        for entry in manifest:
            with open(entry["indices_path"]) as f:
                indices = json.load(f)
            samples.append({
                "image_path": entry["image_path"],
                "indices": indices,
                "raw_label": f"SimSat_{entry['name']}",
                "split": "test" # SimSat for evaluation
            })
            
    logger.info("Loaded %d samples from raw sources", len(samples))
    return samples

# ── Saving ─────────────────────────────────────────────────────────────────────
def save_dataset(samples: List[Dict], output_dir: str = "data/dataset"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    splits = {"train": [], "val": [], "test": []}
    stats = {
        "counts": {"train": {}, "val": {}, "test": {}},
        "indices_stats": {}
    }
    
    for s in samples:
        stress_class, severity, reasoning = label_to_stress_class(s["raw_label"], s["indices"])
        
        # 3 variants
        for i in range(3):
            sys_p, user_p, resp = build_prompt(s["indices"], stress_class, reasoning, i)
            line = {
                "image": s["image_path"],
                "system": sys_p,
                "prompt": user_p,
                "response": resp,
                "label": stress_class,
                "indices": s["indices"]
            }
            splits[s["split"]].append(line)
            
        # Stats update
        split = s["split"]
        stats["counts"][split][stress_class] = stats["counts"][split].get(stress_class, 0) + 3

    # Add Negative Examples to train
    for neg in NEGATIVE_EXAMPLES:
        for i in range(3):
            sys_p, user_p, resp = build_prompt(neg["indices"], neg["label"], neg["reason"], i)
            line = {
                "image": "",
                "system": sys_p,
                "prompt": user_p,
                "response": resp,
                "label": neg["label"],
                "indices": neg["indices"]
            }
            splits["train"].append(line)

    # Write files
    for split, lines in splits.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        logger.info("Wrote %d lines to %s.jsonl", len(lines), split)

    with open(out / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset build complete.")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CropAlert Dataset Builder")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/dataset")
    args = parser.parse_args()
    
    samples = load_all_samples(args.raw_dir)
    if not samples:
        logger.error("No samples found. Run download_datasets.py first.")
        return
        
    save_dataset(samples, args.output_dir)

if __name__ == "__main__":
    main()
