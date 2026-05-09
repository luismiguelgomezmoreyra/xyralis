import pytest
import httpx
import json
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path
from fastapi import FastAPI
from api.main import app, state
from api.inference import AnalysisResult

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_inference():
    """Mock the inference engine to avoid loading 7B model in tests."""
    mock = MagicMock()
    mock.model_version = "test-v1"
    
    # Mock return for analyze
    mock.analyze.return_value = AnalysisResult(
        classification="healthy",
        confidence=0.95,
        recommendation="Maintain current irrigation.",
        days_to_action=10,
        detected_issues=[],
        ndvi_mean=0.7,
        stress_score=5.0,
        inference_latency_ms=120.0,
        model_version="test-v1"
    )
    
    # Mock batch_analyze
    mock.batch_analyze.return_value = [mock.analyze.return_value]
    
    return mock

@pytest.fixture
def mock_simsat():
    """Mock SimSatClient responses."""
    mock = MagicMock()
    mock.fetch_all_views.return_value = {
        "sentinel_png": "data/demo/mantaro_pe.png",
        "mapbox_png": "data/demo/mantaro_pe_mapbox.png",
        "indices": {"ndvi_mean": 0.7, "ndwi_mean": 0.1, "stress_score": 5.0},
        "metadata": {"sensor": "Sentinel-2", "cloud_cover": 0.1}
    }
    return mock

# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(mock_simsat):
    # Ensure state is initialized
    state.inference = MagicMock()
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            response = await ac.get("/health")
            
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["simsat_available"] is True

@pytest.mark.asyncio
async def test_analyze_endpoint(mock_inference, mock_simsat):
    state.inference = mock_inference
    
    payload = {"latitude": -12.05, "longitude": -75.2, "radius_km": 1.0}
    
    with patch("api.main.get_simsat_client", return_value=mock_simsat):
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/analyze", json=payload)
            
    assert response.status_code == 200
    data = response.json()
    assert data["analysis"]["classification"] == "healthy"
    assert "sentinel_png" in data["image_urls"]
    assert data["satellite_metadata"]["sensor"] == "Sentinel-2"

@pytest.mark.asyncio
async def test_batch_analyze_limit(mock_inference, mock_simsat):
    state.inference = mock_inference
    
    # 21 parcels (> 20 limit)
    parcels = [{"latitude": 0, "longitude": 0, "name": f"p{i}"} for i in range(21)]
    payload = {"parcels": parcels}
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/batch_analyze", json=payload)
        
    # Pydantic validates this, returning 422 Unprocessable Entity
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_demo_endpoint(mock_inference):
    state.inference = mock_inference
    
    # Ensure demo file exists for test or mock Path.exists
    with patch("pathlib.Path.exists", return_value=True):
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/demo")
            
    assert response.status_code == 200
    data = response.json()
    assert "Peru" in str(data["satellite_metadata"])

@pytest.mark.asyncio
async def test_simulation_status_proxy():
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, 
                json=lambda: {"satellite_position": {"lat": 10, "lon": 10}, "timestamp": "now", "simulation_speed": 1.0}
            )
            response = await ac.get("/simulation/status")
            
    assert response.status_code == 200
    assert response.json()["simulation_speed"] == 1.0

@pytest.mark.asyncio
async def test_metrics_endpoint():
    state.total_requests = 10
    state.total_latency_ms = 1000.0
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/metrics")
        
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 10
    assert data["avg_latency_ms"] == 100.0

@pytest.mark.asyncio
async def test_image_serving(mock_simsat):
    # Mocking FileResponse is complex, but we can verify endpoint logic
    with patch("api.main.get_simsat_client", return_value=mock_simsat):
        with patch("api.main.FileResponse") as mock_file_resp:
            mock_file_resp.return_value = MagicMock(status_code=200)
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.get("/images/mapbox/current?lat=0&lon=0")
                
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_cache_logic(mock_inference, mock_simsat):
    state.inference = mock_inference
    state.memory_cache = {} # Reset
    state.cache_hits = 0
    state.cache_misses = 0
    
    payload = {"latitude": 10.0, "longitude": 20.0}
    
    with patch("api.main.get_simsat_client", return_value=mock_simsat):
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            # First call: Miss
            await ac.post("/analyze", json=payload)
            assert state.cache_misses == 1
            
            # Second call: Hit
            await ac.post("/analyze", json=payload)
            assert state.cache_hits == 1
