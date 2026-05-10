"""
Xyralis Unified API - Satellite Data Processing & AI Inference
Consolidated service for Sentinel-2 processing and Liquid AI analysis.
No simulations, no mocks. 100% Real data.
"""

import asyncio
import json
import logging
import os
import time
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import httpx
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from api.inference import XyralisInference, AnalysisResult
from api.schemas import (
    AnalyzeRequest, AnalyzeResponse, BatchAnalyzeRequest, BatchAnalyzeResponse,
    HealthResponse, SimulationStatus, SimulationControl, MetricsResponse,
    BatchSummary, AnalysisResultSchema
)

load_dotenv()

# Setup Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "xyralis.log", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("xyralis.app")

# ── Global State & Metrics ─────────────────────────────────────────────────────
class GlobalState:
    def __init__(self):
        self.inference: Optional[XyralisInference] = None
        self.start_time = time.time()
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.class_counts = {"healthy": 0, "mild_stress": 0, "severe_stress": 0, "uncertain": 0}
        self.memory_cache: Dict[str, Dict] = {}

state = GlobalState()

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model on Startup
    model_path = os.getenv("MODEL_PATH", "training/weights/fast_model/xyralis-lora")
    try:
        if not Path(model_path).exists():
             logger.warning(f"Model path {model_path} does not exist. Inference will use CPU fallback.")
        state.inference = XyralisInference(model_path)
        logger.info("Xyralis Inference Engine started successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    logger.info("Shutting down Xyralis API.")

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Xyralis API",
    description="Procesamiento de imágenes Sentinel-2 y detección de estrés agrícola con IA Real",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/data/images", StaticFiles(directory="data/images"), name="data_images")

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_real_data_for_coords(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Finds the closest local Sentinel-2 product for given coordinates."""
    dataset_path = Path("data/processed/dataset_labels.json")
    if not dataset_path.exists():
        return None
        
    with open(dataset_path) as f:
        dataset = json.load(f)
        
    if not dataset:
        return None
        
    # Simple distance search
    def dist(p):
        return (p.get("lat", 0) - lat)**2 + (p.get("lon", 0) - lon)**2
        
    closest = min(dataset, key=dist)
    
    # Threshold check: only use if within ~100km (approx 1 degree)
    if dist(closest) > 1.0:
        return None
        
    img_name = f"{closest['filename']}_{closest['label']}.png"
    img_path = Path("data/images") / img_name
    
    return {
        "image_path": str(img_path),
        "indices": closest["indices"],
        "metadata": {
            "product_id": closest["filename"],
            "lat": closest.get("lat"),
            "lon": closest.get("lon"),
            "source": "Sentinel-2 L1C/L2A"
        }
    }

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": state.inference is not None,
        "simsat_available": False,
        "uptime_seconds": time.time() - state.start_time,
        "model_version": state.inference.model_version if state.inference else "none"
    }

@app.get("/all_parcels")
async def get_all_parcels():
    """Returns all real processed sentinel parcels."""
    dataset_path = Path("data/processed/dataset_labels.json")
    if not dataset_path.exists():
        return []
    with open(dataset_path) as f:
        return json.load(f)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_parcel(req: AnalyzeRequest):
    state.total_requests += 1
    
    # 1. Look for REAL local data
    data = get_real_data_for_coords(req.latitude, req.longitude)
    if not data:
        raise HTTPException(status_code=404, detail="No real Sentinel-2 data available for these coordinates. Please click on a green marker.")
    
    try:
        # 2. IA Inference on REAL image
        result: AnalysisResult = state.inference.analyze(data["image_path"], data["indices"])
        
        state.total_latency_ms += result.inference_latency_ms
        state.class_counts[result.classification] += 1
        
        return {
            "analysis": result.__dict__,
            "image_urls": {
                "sentinel_png": f"/data/images/{Path(data['image_path']).name}",
                "mapbox_png": "" # Real mapbox requires key, using Leaflet instead
            },
            "satellite_metadata": data["metadata"]
        }

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"IA Engine Error: {str(e)}")

@app.get("/demo")
async def run_random_real_demo():
    """Selects a random real product and analyzes it."""
    dataset_path = Path("data/processed/dataset_labels.json")
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="No data processed.")
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    sample = dataset[np.random.randint(0, len(dataset))]
    return await analyze_parcel(AnalyzeRequest(latitude=sample["lat"], longitude=sample["lon"]))

# --- Maintenance Endpoints (Real actions) ---

@app.post("/unpack")
async def unpack_new_data():
    from data.unpack_sentinel2 import unpack_sentinel2_zips
    extracted = unpack_sentinel2_zips("data/raw/sentinel2")
    return {"status": "success", "extracted": extracted}

@app.post("/compute")
async def recompute_indices():
    from data.compute_indices import process_dataset
    from data.add_coords import update_labels
    process_dataset("data/raw/sentinel2", "data/processed/dataset_labels.json")
    update_labels() # Re-extract coords
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
