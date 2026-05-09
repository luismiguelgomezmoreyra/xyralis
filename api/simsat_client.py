"""
SimSat API Client for CropAlert — production-quality module.
Handles satellite image retrieval, spectral index computation, and demo caching.
"""

from __future__ import annotations

import os
import io
import json
import time
import logging
import requests
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("cropalert.simsat")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(_h)

# ── Environment ─────────────────────────────────────────────────────────────
SIMSAT_URL = os.getenv("SIMSAT_URL", "http://localhost:9005")

# ── Custom Exceptions ───────────────────────────────────────────────────────
class SimSatConnectionError(RuntimeError):
    """Raised when the SimSat API is unreachable after retries."""

class SimSatImageUnavailableError(RuntimeError):
    """Raised when an image endpoint returns image_available=False."""

class SimSatBandError(RuntimeError):
    """Raised when a required spectral band is missing or malformed."""

# ── Dataclasses ─────────────────────────────────────────────────────────────
@dataclass
class SentinelResult:
    image_available: bool
    cloud_cover: Optional[float] = None
    source: Optional[str] = None
    footprint: Optional[str] = None
    datetime: Optional[str] = None
    bands_array: Optional[np.ndarray] = None  # shape [6, H, W]
    false_color_png_path: Optional[str] = None
    indices: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class MapboxResult:
    target_visible: bool
    image_available: bool
    elevation_degrees: Optional[float] = None
    bearing: Optional[float] = None
    pitch: Optional[float] = None
    satellite_position: Optional[dict] = None
    timestamp: Optional[str] = None
    png_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AllViewsResult:
    sentinel_historical: Optional[SentinelResult] = None
    sentinel_live: Optional[SentinelResult] = None
    mapbox_live: Optional[MapboxResult] = None
    mapbox_fixed: Optional[MapboxResult] = None
    best_image_path: Optional[str] = None
    indices: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ── SimSatClient ─────────────────────────────────────────────────────────────
class SimSatClient:
    """Production-grade client for SimSat satellite simulation API."""

    BASE: str = SIMSAT_URL
    TIMEOUT: int = 120  # Sentinel endpoints can be slow
    RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.5

    # Band order expected by index computation
    BAND_ORDER = ["red", "green", "blue", "nir", "swir16", "swir22"]
    DEFAULT_BANDS = ["red", "green", "blue", "nir", "swir16", "swir22"]

    # ─── Position ──────────────────────────────────────────────────────────────
    def get_current_position(self) -> dict:
        """GET /data/current/position with retry."""
        url = f"{self.BASE}/data/current/position"
        for attempt in range(1, self.RETRIES + 1):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "lon": data["longitude_deg"],
                    "lat": data["latitude_deg"],
                    "alt": data.get("altitude_km"),
                    "timestamp": data.get("timestamp_iso"),
                }
            except (requests.RequestException, KeyError) as exc:
                logger.warning("get_current_position attempt %d/%d failed: %s", attempt, self.RETRIES, exc)
                if attempt == self.RETRIES:
                    raise SimSatConnectionError(f"Position fetch failed: {exc}") from exc
                time.sleep(self.BACKOFF_FACTOR ** attempt)

    # ─── Sentinel historical ──────────────────────────────────────────────────
    def fetch_current_sentinel(
        self,
        size_km: float = 5.0,
        window_seconds: int = 864000,
        bands: Optional[List[str]] = None,
    ) -> SentinelResult:
        """GET /data/current/image/sentinel (array return_type)."""
        bands = bands or self.DEFAULT_BANDS
        params = {
            "size_km": size_km,
            "window_seconds": window_seconds,
            "bands": ",".join(bands),
            "return_type": "array",
        }
        url = f"{self.BASE}/data/current/image/sentinel"
        return self._fetch_sentinel_common(url, params)

    def fetch_sentinel_at(
        self,
        lat: float,
        lon: float,
        timestamp_iso: str,
        size_km: float = 5.0,
        window_seconds: int = 864000,
    ) -> SentinelResult:
        """GET /data/image/sentinel for a specific historical point."""
        params = {
            "lat": lat,
            "lon": lon,
            "timestamp": timestamp_iso,
            "size_km": size_km,
            "window_seconds": window_seconds,
            "return_type": "array",
        }
        url = f"{self.BASE}/data/image/sentinel"
        return self._fetch_sentinel_common(url, params)

    def _fetch_sentinel_common(self, url: str, params: dict) -> SentinelResult:
        """Shared logic for both Sentinel endpoints."""
        for attempt in range(1, self.RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.TIMEOUT)
                resp.raise_for_status()
                payload = resp.json()

                result = SentinelResult(
                    image_available=payload.get("image_available", False),
                    cloud_cover=payload.get("cloud_cover"),
                    source=payload.get("source"),
                    footprint=payload.get("footprint"),
                    datetime=payload.get("datetime"),
                )

                if not result.image_available:
                    logger.info("Sentinel image not available (cloud_cover=%s)", result.cloud_cover)
                    return result

                # bands_array: [6, H, W] float32 normalized 0-1
                bands_list = payload.get("bands", [])
                if len(bands_list) != 6:
                    raise SimSatBandError(f"Expected 6 bands, got {len(bands_list)}")
                result.bands_array = np.array(bands_list, dtype=np.float32)

                # Compute indices + false-color composite
                ts = int(time.time())
                result.indices = compute_spectral_indices(
                    result.bands_array,
                    output_colormap_path=self._demo_path(f"ndvi_colormap_{ts}.png")
                )
                result.false_color_png_path = make_false_color_composite(
                    result.bands_array,
                    output_path=self._demo_path(f"sentinel_fc_{ts}.png"),
                )
                return result

            except requests.RequestException as exc:
                logger.warning("_fetch_sentinel_common attempt %d/%d failed: %s", attempt, self.RETRIES, exc)
                if attempt == self.RETRIES:
                    return SentinelResult(image_available=False, error=str(exc))
                time.sleep(self.BACKOFF_FACTOR ** attempt)

    # ─── Mapbox live ──────────────────────────────────────────────────────────
    def fetch_current_mapbox(
        self,
        target_lat: Optional[float] = None,
        target_lon: Optional[float] = None,
    ) -> MapboxResult:
        """GET /data/current/image/mapbox."""
        params = {}
        if target_lat is not None:
            params["target_lat"] = target_lat
        if target_lon is not None:
            params["target_lon"] = target_lon

        url = f"{self.BASE}/data/current/image/mapbox"
        return self._fetch_mapbox_common(url, params)

    def fetch_mapbox_at(
        self,
        lat_target: float,
        lon_target: float,
        lat_sat: float,
        lon_sat: float,
        alt_sat_km: float,
    ) -> MapboxResult:
        """GET /data/image/mapbox for fixed positions."""
        params = {
            "lon_target": lon_target,
            "lat_target": lat_target,
            "lon_satellite": lon_sat,
            "lat_satellite": lat_sat,
            "alt_satellite": alt_sat_km,
        }
        url = f"{self.BASE}/data/image/mapbox"
        return self._fetch_mapbox_common(url, params)

    def _fetch_mapbox_common(self, url: str, params: dict) -> MapboxResult:
        """Shared logic for both Mapbox endpoints."""
        for attempt in range(1, self.RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.TIMEOUT)
                resp.raise_for_status()
                payload = resp.json()

                result = MapboxResult(
                    target_visible=payload.get("target_visible", False),
                    image_available=payload.get("image_available", False),
                    elevation_degrees=payload.get("elevation_degrees"),
                    bearing=payload.get("bearing_deg"),
                    pitch=payload.get("pitch_deg"),
                    satellite_position=payload.get("satellite_position"),
                    timestamp=payload.get("timestamp_iso"),
                )

                if not result.target_visible:
                    logger.info("Mapbox target not visible (elevation=%s°)", result.elevation_degrees)
                    return result

                if result.image_available:
                    # Download PNG
                    img_url = payload.get("image_url")
                    if img_url:
                        img_resp = requests.get(img_url, timeout=self.TIMEOUT)
                        img_resp.raise_for_status()
                        png_path = self._demo_path(f"mapbox_{int(time.time())}.png")
                        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
                        Image.open(io.BytesIO(img_resp.content)).save(png_path)
                        result.png_path = png_path

                return result

            except requests.RequestException as exc:
                logger.warning("_fetch_mapbox_common attempt %d/%d failed: %s", attempt, self.RETRIES, exc)
                if attempt == self.RETRIES:
                    return MapboxResult(target_visible=False, image_available=False, error=str(exc))
                time.sleep(self.BACKOFF_FACTOR ** attempt)

    # ─── Parallel fetch all ───────────────────────────────────────────────────
    def fetch_all_views(
        self,
        lat: float,
        lon: float,
        timestamp_iso: str,
    ) -> AllViewsResult:
        """
        Calls all 4 image endpoints concurrently:
        - sentinel_historical (current)
        - sentinel_live (at timestamp)
        - mapbox_live (current)
        - mapbox_fixed (at timestamp + position)
        """
        result = AllViewsResult()
        errors: List[str] = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.fetch_current_sentinel): "sentinel_historical",
                executor.submit(self.fetch_sentinel_at, lat, lon, timestamp_iso): "sentinel_live",
                executor.submit(self.fetch_current_mapbox, lat, lon): "mapbox_live",
                executor.submit(self.fetch_mapbox_at, lat, lon, lat, lon, 700.0): "mapbox_fixed",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    setattr(result, name, res)
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                    logger.error("fetch_all_views – %s failed: %s", name, exc)

        result.errors = errors

        # Pick best image (priority: sentinel false-color)
        sentinel = result.sentinel_historical or result.sentinel_live
        if sentinel and sentinel.false_color_png_path:
            result.best_image_path = sentinel.false_color_png_path
        elif result.mapbox_live and result.mapbox_live.png_path:
            result.best_image_path = result.mapbox_live.png_path
        elif result.mapbox_fixed and result.mapbox_fixed.png_path:
            result.best_image_path = result.mapbox_fixed.png_path

        # Merge indices
        if sentinel and sentinel.indices:
            result.indices = sentinel.indices

        return result

    # ─── Demo cache ───────────────────────────────────────────────────────────
    def cache_demo_images(self, parcels: List[dict]) -> List[str]:
        """
        Pre-fetch and cache all views for a list of parcels.
        Returns list of cached image paths.
        """
        cached: List[str] = []
        manifest: List[dict] = []
        demo_dir = Path("data/demo")
        demo_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Caching demo images for %d parcels…", len(parcels))

        for parcel in parcels:
            lat = parcel["lat"]
            lon = parcel["lon"]
            ts = parcel["timestamp"]
            name = parcel.get("name", f"{lat:.4f}_{lon:.4f}")

            logger.info("Fetching parcel %s …", name)
            views = self.fetch_all_views(lat, lon, ts)

            # Save each available image
            for attr in ["sentinel_historical", "sentinel_live", "mapbox_live", "mapbox_fixed"]:
                res = getattr(views, attr)
                if res and res.false_color_png_path and Path(res.false_color_png_path).exists():
                    dest = demo_dir / f"{name}_{attr}.png"
                    Path(res.false_color_png_path).rename(dest)
                    cached.append(str(dest))
                    manifest.append({"parcel": name, "view": attr, "path": str(dest)})
                elif res and res.png_path and Path(res.png_path).exists():
                    dest = demo_dir / f"{name}_{attr}.png"
                    Path(res.png_path).rename(dest)
                    cached.append(str(dest))
                    manifest.append({"parcel": name, "view": attr, "path": str(dest)})

            # Save indices JSON
            if views.indices:
                idx_path = demo_dir / f"{name}_indices.json"
                with open(idx_path, "w") as f:
                    json.dump(views.indices, f, indent=2)
                manifest.append({"parcel": name, "view": "indices", "path": str(idx_path)})

        # Write manifest
        with open(demo_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Cached %d images for %d parcels", len(cached), len(parcels))
        return cached

    # ─── Helper ───────────────────────────────────────────────────────────────
    def _demo_path(self, filename: str) -> str:
        """Return path inside data/demo."""
        p = Path("data/demo") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)


# ── Index computation ─────────────────────────────────────────────────────────
def compute_spectral_indices(bands_array: np.ndarray, output_colormap_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute NDVI, NDWI, SWIR_ratio, EVI + statistics.
    bands_array: [6, H, W] in order [red, green, blue, nir, swir16, swir22]
    All bands normalized 0-1 float32.

    Returns dict with _mean, _std, _min, _max for each index, stress_score,
    and optionally ndvi_colormap_path if output_colormap_path is provided.
    """
    if bands_array.ndim != 3 or bands_array.shape[0] != 6:
        raise SimSatBandError(f"Expected shape [6,H,W], got {bands_array.shape}")

    red   = bands_array[0]
    green = bands_array[1]
    blue  = bands_array[2]
    nir   = bands_array[3]
    sw16  = bands_array[4]
    sw22  = bands_array[5]

    eps = 1e-7

    ndvi  = (nir - red) / (nir + red + eps)
    ndwi  = (nir - sw16) / (nir + sw16 + eps)
    swir  = sw16 / (sw22 + eps)
    evi   = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)

    # Clamp to valid ranges before stats
    ndvi = np.clip(ndvi, -1, 1)
    ndwi = np.clip(ndwi, -1, 1)
    evi  = np.clip(evi, 0, 3)

    def stats(arr: np.ndarray) -> Dict[str, float]:
        flat = arr.flatten()
        return {
            "mean": float(np.nanmean(flat)),
            "std":  float(np.nanstd(flat)),
            "min":  float(np.nanmin(flat)),
            "max":  float(np.nanmax(flat)),
        }

    ndvi_stats = stats(ndvi)
    ndwi_stats = stats(ndwi)
    swir_stats = stats(swir)
    evi_stats  = stats(evi)

    # Stress score: 0 = healthy, 100 = severely stressed
    water_stress = 100 * (1.0 - float(np.clip((ndwi_stats["mean"] + 0.2) / 0.5, 0, 1)))
    veget_stress = 100 * (1.0 - float(np.clip(ndvi_stats["mean"], 0, 1)))
    stress_score = 0.6 * veget_stress + 0.4 * water_stress

    result = {
        **{f"ndvi_{k}": v for k, v in ndvi_stats.items()},
        **{f"ndwi_{k}": v for k, v in ndwi_stats.items()},
        **{f"swir_ratio_{k}": v for k, v in swir_stats.items()},
        **{f"evi_{k}": v for k, v in evi_stats.items()},
        "stress_score": float(np.clip(stress_score, 0, 100)),
    }

    # Optional NDVI colormap
    if output_colormap_path:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("viridis")
            ndvi_rgba = cmap(np.clip(ndvi, -1, 1))  # [H,W,4] in 0-1
            ndvi_rgb = (ndvi_rgba[..., :3] * 255).astype(np.uint8)
            img = Image.fromarray(ndvi_rgb)
            Path(output_colormap_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_colormap_path)
            result["ndvi_colormap_path"] = output_colormap_path
        except Exception as e:
            logger.warning("Failed to generate NDVI colormap: %s", e)

    return result


# ── False-colour composite ────────────────────────────────────────────────────
def make_false_color_composite(
    bands_array: np.ndarray,
    output_path: str,
    size: Tuple[int, int] = (224, 224),
) -> str:
    """
    Build NIR-Red-Green false-colour PNG.
    R channel ← NIR (index 3)
    G channel ← Red (index 0)
    B channel ← Green (index 1)
    Normalized by 2nd–98th percentiles to suppress outliers.
    """
    if bands_array.ndim != 3 or bands_array.shape[0] < 4:
        raise SimSatBandError("bands_array must have at least 4 bands [R,G,B,NIR,…]")

    nir  = bands_array[3]  # NIR
    red = bands_array[0]  # Red
    grn = bands_array[1]  # Green

    rgb = np.stack([nir, red, grn], axis=-1)  # [H, W, 3]

    # Robust percentile normalisation
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb).resize(size, Image.LANCZOS)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.debug("False-colour saved: %s", output_path)
    return output_path


# ── Demo ─────────────────────────────────────────────────────────────────────
def _demo_parcels() -> List[dict]:
    """Default demo parcels around Valle del Mantaro, Perú."""
    return [
        {"lat": -12.0, "lon": -75.2, "timestamp": "2026-03-01T12:00:00Z", "name": "parcel_A"},
        {"lat": -12.1, "lon": -75.3, "timestamp": "2026-03-01T12:00:00Z", "name": "parcel_B"},
    ]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CropAlert — SimSat Client Test Suite")
    print("=" * 60)

    client = SimSatClient()
    summary = {
        "get_current_position": "PENDING",
        "fetch_current_sentinel": "PENDING",
        "fetch_current_mapbox": "PENDING",
        "fetch_sentinel_at": "PENDING",
        "fetch_mapbox_at": "PENDING",
        "fetch_all_views": "PENDING",
        "cache_demo_images": "PENDING",
    }

    try:
        # 1. Position
        print("\n[1/7] get_current_position()")
        pos = client.get_current_position()
        print(f"  lon={pos['lon']:.6f} lat={pos['lat']:.6f} alt={pos['alt']}km")
        summary["get_current_position"] = "OK"

    except Exception as e:
        logger.error("get_current_position failed: %s", e)
        summary["get_current_position"] = f"FAIL — {e}"

    try:
        # 2. Current Sentinel
        print("\n[2/7] fetch_current_sentinel()")
        sent = client.fetch_current_sentinel(size_km=5.0)
        if sent.image_available:
            print(f"  Cloud cover: {sent.cloud_cover:.1f}%")
            print(f"  Bands shape : {sent.bands_array.shape}")
            print(f"  Indices     : NDVI={sent.indices.get('ndvi_mean', 'n/a'):.3f}")
            print(f"  False-color : {sent.false_color_png_path}")
            if 'ndvi_colormap_path' in sent.indices:
                print(f"  NDVI map    : {sent.indices['ndvi_colormap_path']}")
            summary["fetch_current_sentinel"] = "OK"
        else:
            print("  Image not available (clouds / no coverage)")
            summary["fetch_current_sentinel"] = "NO IMAGE"

    except Exception as e:
        logger.error("fetch_current_sentinel failed: %s", e)
        summary["fetch_current_sentinel"] = f"FAIL — {e}"

    try:
        # 3. Current Mapbox
        print("\n[3/7] fetch_current_mapbox()")
        mbx = client.fetch_current_mapbox()
        if mbx.target_visible:
            print(f"  Elevation   : {mbx.elevation_degrees:.1f}°")
            print(f"  Image       : {mbx.png_path}")
            summary["fetch_current_mapbox"] = "OK"
        else:
            print(f"  Target not visible (elev {mbx.elevation_degrees}°)")
            summary["fetch_current_mapbox"] = "NOT VISIBLE"

    except Exception as e:
        logger.error("fetch_current_mapbox failed: %s", e)
        summary["fetch_current_mapbox"] = f"FAIL — {e}"

    try:
        # 4. Historical Sentinel
        print("\n[4/7] fetch_sentinel_at()")
        sent_hist = client.fetch_sentinel_at(
            lat=-12.0, lon=-75.2, timestamp_iso="2026-03-01T12:00:00Z"
        )
        if sent_hist.image_available:
            print(f"  NDVI        : {sent_hist.indices.get('ndvi_mean', 'n/a'):.3f}")
            summary["fetch_sentinel_at"] = "OK"
        else:
            print("  Historical image not available")
            summary["fetch_sentinel_at"] = "NO IMAGE"

    except Exception as e:
        logger.error("fetch_sentinel_at failed: %s", e)
        summary["fetch_sentinel_at"] = f"FAIL — {e}"

    try:
        # 5. Fixed Mapbox
        print("\n[5/7] fetch_mapbox_at()")
        mbx_fixed = client.fetch_mapbox_at(
            lat_target=-12.0, lon_target=-75.2, lat_sat=-12.0, lon_sat=-75.2, alt_sat_km=700.0
        )
        if mbx_fixed.image_available:
            print(f"  PNG path    : {mbx_fixed.png_path}")
            summary["fetch_mapbox_at"] = "OK"
        else:
            print("  Fixed Mapbox image not available")
            summary["fetch_mapbox_at"] = "NO IMAGE"

    except Exception as e:
        logger.error("fetch_mapbox_at failed: %s", e)
        summary["fetch_mapbox_at"] = f"FAIL — {e}"

    try:
        # 6. All views parallel
        print("\n[6/7] fetch_all_views()")
        all_views = client.fetch_all_views(lat=-12.0, lon=-75.2, timestamp_iso="2026-03-01T12:00:00Z")
        print(f"  Errors      : {len(all_views.errors)}")
        print(f"  Best image  : {all_views.best_image_path}")
        if all_views.indices:
            print(f"  NDVI mean   : {all_views.indices.get('ndvi_mean', 'n/a'):.3f}")
            if 'ndvi_colormap_path' in all_views.indices:
                print(f"  NDVI map    : {all_views.indices['ndvi_colormap_path']}")
        summary["fetch_all_views"] = "OK"

    except Exception as e:
        logger.error("fetch_all_views failed: %s", e)
        summary["fetch_all_views"] = f"FAIL — {e}"

    try:
        # 7. Demo cache
        print("\n[7/7] cache_demo_images()")
        cached = client.cache_demo_images(_demo_parcels())
        print(f"  Cached %d images", len(cached))
        summary["cache_demo_images"] = "OK"

    except Exception as e:
        logger.error("cache_demo_images failed: %s", e)
        summary["cache_demo_images"] = f"FAIL — {e}"

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, status in summary.items():
        print(f"  {key:<30s} → {status}")
    print("=" * 60)

    total = len(summary)
    ok = sum(1 for s in summary.values() if s == "OK")
    print(f"\nTotal: {ok}/{total} tests passed\n")
    sys.exit(0 if ok == total else 1)
