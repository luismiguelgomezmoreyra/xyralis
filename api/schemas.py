from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# ── Inference Schemas ──────────────────────────────────────────────────────────

class AnalysisResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    classification: Literal["healthy", "mild_stress", "severe_stress", "uncertain"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level between 0.0 and 1.0")
    recommendation: str
    days_to_action: int = Field(..., description="Estimated days until intervention is required. -1 indicates uncertainty.")
    detected_issues: List[str]
    ndvi_mean: float
    stress_score: float = Field(..., ge=0.0, le=100.0)
    inference_latency_ms: float
    model_version: str

# ── API Request/Response Schemas ───────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = 1.0
    timestamp: Optional[str] = None  # ISO format

class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    analysis: AnalysisResultSchema
    image_urls: Dict[str, str]
    satellite_metadata: Dict[str, str | float | int]

class ParcelItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    latitude: float
    longitude: float
    name: str

class BatchAnalyzeRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    parcels: List[ParcelItem] = Field(..., max_length=20)
    timestamp: Optional[str] = None

class BatchSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    total_count: int
    class_distribution: Dict[str, int]
    mean_stress_score: float

class BatchAnalyzeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    results: List[AnalysisResultSchema]
    summary_stats: BatchSummary

class HealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    status: str
    model_loaded: bool
    simsat_available: bool
    uptime_seconds: float
    model_version: str

class SimulationStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    satellite_position: Dict[str, float]
    timestamp: str
    simulation_speed: float

class SimulationControl(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    command: Literal["start", "pause", "stop"]
    start_time: Optional[str] = None
    step_size_seconds: Optional[int] = None
    replay_speed: Optional[float] = None

class MetricsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    total_requests: int
    avg_latency_ms: float
    classification_distribution: Dict[str, int]
    cache_hit_rate: float
