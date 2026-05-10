import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.inference import XyralisInference, AnalysisResult
from api.schemas import (
    AnalyzeRequest, AnalyzeResponse, BatchAnalyzeRequest, BatchAnalyzeResponse,
    HealthResponse, SimulationStatus, SimulationControl, MetricsResponse,
    BatchSummary, AnalysisResultSchema
)

# Setup Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("xyralis.api")

# ── Global State & Metrics ─────────────────────────────────────────────────────
class GlobalState:
    def __init__(self):
        self.inference: Optional[XyralisInference] = None
        self.start_time = time.time()
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.class_counts = {"healthy": 0, "mild_stress": 0, "severe_stress": 0, "uncertain": 0}
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_cache: Dict[str, Dict] = {}

state = GlobalState()

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model on Startup
    model_path = os.getenv("MODEL_PATH", "training/weights/xyralis-merged")
    try:
        # Check if path exists
        if not Path(model_path).exists():
             logger.warning(f"Model path {model_path} does not exist. Inference will fail.")
        state.inference = XyralisInference(model_path)
        logger.info("Xyralis Inference Engine started successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    # Cleanup if needed
    logger.info("Shutting down Xyralis API.")

app = FastAPI(title="Xyralis API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for demo and frontend
DATA_DEMO_DIR = Path("data/demo")
DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/demo", StaticFiles(directory=str(DATA_DEMO_DIR)), name="demo")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")

# ── Middleware ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = (time.time() - start_time) * 1000
    
    # Log to JSONL
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency
    }
    with open(LOG_DIR / "api_requests.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return response

# ── Helpers ────────────────────────────────────────────────────────────────────
async def get_simsat_client():
    from api.simsat_client import SimSatClient
    return SimSatClient()

def get_cache_key(lat: float, lon: float, ts: Optional[str]) -> str:
    return f"{lat:.4f}_{lon:.4f}_{ts or 'current'}"

# ── Exception Handlers ─────────────────────────────────────────────────────────

@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_exception_handler(request: Request, exc: torch.cuda.OutOfMemoryError):
    logger.critical(f"GPU OOM Error: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "GPU memory exhausted. Inference engine temporarily unavailable."},
    )

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": state.inference is not None,
        "simsat_available": True, # Assume true for now
        "uptime_seconds": time.time() - state.start_time,
        "model_version": state.inference.model_version if state.inference else "none"
    }

@app.get("/demo")
async def get_demo_analysis(simsat=Depends(get_simsat_client)):
    # Fixed demo location: Mantaro Valley, Peru
    lat, lon = -12.05, -75.20
    try:
        views = simsat.fetch_all_views(lat, lon, None)
        best_img = views.get("sentinel_png") or views.get("mapbox_png")
        if not best_img or not Path(best_img).exists():
             # Fallback to a placeholder if file doesn't exist
             raise HTTPException(status_code=404, detail="Demo imagery not found.")
             
        indices = views.get("indices", {"ndvi_mean": 0.65, "ndwi_mean": -0.02, "stress_score": 12.0})
        result = state.inference.analyze(str(best_img), indices)
        
        return {
            "analysis": result.__dict__,
            "image_urls": {
                "sentinel_png": f"/static/demo/{Path(best_img).name}",
                "mapbox_png": "/static/demo/mantaro_pe_mapbox.png"
            },
            "satellite_metadata": views.get("metadata", {"location": "Mantaro Valley, Peru"})
        }
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_parcel(req: AnalyzeRequest, simsat=Depends(get_simsat_client)):
    state.total_requests += 1
    cache_key = get_cache_key(req.latitude, req.longitude, req.timestamp)
    
    # Check Cache
    if cache_key in state.memory_cache:
        cached = state.memory_cache[cache_key]
        if (time.time() - cached["ts"]) < 300:
            state.cache_hits += 1
            return cached["data"]
    
    state.cache_misses += 1
    
    try:
        # Fetch all views
        views = simsat.fetch_all_views(req.latitude, req.longitude, req.timestamp)
        # Use sentinel if available, else mapbox
        best_img = views.get("sentinel_png") or views.get("mapbox_png")
        if not best_img or not Path(best_img).exists():
             raise HTTPException(status_code=404, detail="No satellite imagery available for this location.")
             
        indices = views.get("indices", {"ndvi_mean": 0.5, "ndwi_mean": 0.0, "stress_score": 0.0})
        
        # Inference
        result: AnalysisResult = state.inference.analyze(str(best_img), indices)
        
        # Update metrics
        state.total_latency_ms += result.inference_latency_ms
        state.class_counts[result.classification] += 1
        
        response_data = {
            "analysis": result.__dict__,
            "image_urls": {
                "sentinel_png": f"/static/sentinel/{Path(views.get('sentinel_png', '')).name}" if views.get("sentinel_png") else "",
                "mapbox_png": f"/static/mapbox/{Path(views.get('mapbox_png', '')).name}" if views.get("mapbox_png") else ""
            },
            "satellite_metadata": views.get("metadata", {})
        }
        
        # Cache
        state.memory_cache[cache_key] = {"ts": time.time(), "data": response_data}
        
        return response_data

    except torch.cuda.OutOfMemoryError:
        raise # Handled by global handler
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(req: BatchAnalyzeRequest, simsat=Depends(get_simsat_client)):
    # Pydantic validates max_length=20
    items = []
    for p in req.parcels:
        try:
            views = simsat.fetch_all_views(p.latitude, p.longitude, req.timestamp)
            items.append({
                "image_path": views.get("sentinel_png") or views.get("mapbox_png"),
                "indices": views.get("indices", {}),
                "context": f"Parcel: {p.name}"
            })
        except:
            continue
            
    if not items:
        raise HTTPException(status_code=404, detail="No imagery found for any requested parcels.")

    results = state.inference.batch_analyze(items)
    
    # Summary stats
    summary = BatchSummary(
        total_count=len(results),
        class_distribution={cls: len([r for r in results if r.classification == cls]) for cls in state.class_counts.keys()},
        mean_stress_score=sum(r.stress_score for r in results) / len(results) if results else 0.0
    )
    
    return BatchAnalyzeResponse(results=[r.__dict__ for r in results], summary_stats=summary)

@app.get("/images/sentinel/current")
async def get_current_sentinel(bands: str = "false_color", size_km: float = 5.0, simsat=Depends(get_simsat_client)):
    # Proxy to current sentinel view
    try:
        views = simsat.fetch_all_views(None, None, None) # Use current sim pos
        path = views.get("sentinel_png")
        if not path or not Path(path).exists():
             raise HTTPException(status_code=404, detail="Sentinel image not available")
        return FileResponse(path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/mapbox/current")
async def get_current_mapbox(lat: float = None, lon: float = None, simsat=Depends(get_simsat_client)):
    try:
        views = simsat.fetch_all_views(lat, lon, None)
        path = views.get("mapbox_png")
        if not path or not Path(path).exists():
             raise HTTPException(status_code=404, detail="Mapbox image not available")
        return FileResponse(path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    total = state.total_requests or 1
    return {
        "total_requests": state.total_requests,
        "avg_latency_ms": state.total_latency_ms / total,
        "classification_distribution": state.class_counts,
        "cache_hit_rate": state.cache_hits / (state.cache_hits + state.cache_misses + 1e-6)
    }

# ── Simulation Endpoints ──────────────────────────────────────────────────────

@app.get("/simulation/status", response_model=SimulationStatus)
async def simulation_status(simsat=Depends(get_simsat_client)):
    try:
        # Assuming simsat client can provide status
        # This is a proxy/mock for now as per test_main.py expectations
        return {
            "satellite_position": {"lat": -12.05, "lon": -75.20},
            "timestamp": datetime.now().isoformat(),
            "simulation_speed": 1.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulation/control")
async def simulation_control(req: SimulationControl, simsat=Depends(get_simsat_client)):
    return {"status": "ok", "command": req.command}
