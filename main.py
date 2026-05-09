"""
API principal de Xyralis - Procesamiento de Sentinel-2 para agricultura.
Endpoints para descarga, procesamiento y generación de imágenes.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Xyralis API",
    description="Procesamiento de imágenes Sentinel-2 para detección de estrés agrícola",
    version="1.0.0"
)


# === MODELOS ===
class DownloadRequest(BaseModel):
    footprint: Optional[str] = None
    start_date: str = "20240601"
    end_date: str = "20241001"
    max_cloud_cover: int = 20
    product_type: str = "S2MSI2A"


class ProcessRequest(BaseModel):
    raw_dir: str = "data/raw/sentinel2"
    output_json: str = "data/processed/dataset_labels.json"


class GenerateRequest(BaseModel):
    raw_dir: str = "data/raw/sentinel2"
    output_dir: str = "data/images"
    dataset_json: str = "data/processed/dataset_labels.json"
    image_size: tuple = (224, 224)


class PipelineRequest(BaseModel):
    footprint: Optional[str] = None
    start_date: str = "20240601"
    end_date: str = "20241001"
    max_cloud_cover: int = 20


# === ENDPOINTS DE ESTADO ===
@app.get("/")
async def root():
    return {
        "service": "Xyralis Sentinel-2 Processor",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Verifica que las dependencias estén instaladas."""
    try:
        import rasterio
        import numpy
        from PIL import Image
        import sentinelsat

        return {
            "status": "healthy",
            "rasterio": rasterio.__version__,
            "numpy": numpy.__version__,
            "pillow": Image.__version__,
            "sentinelsat": sentinelsat.__version__,
            "copernicus_credentials": bool(os.getenv("COPERNICUS_USER"))
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")


# === ENDPOINTS DE DESCARGA ===
@app.post("/download")
async def download_sentinel_images(request: DownloadRequest):
    """
    Descarga imágenes Sentinel-2 desde Copernicus Hub.
    """
    try:
        from data.download_sentinel2 import download_sentinel2

        result = download_sentinel2(
            footprint=request.footprint,
            start_date=request.start_date,
            end_date=request.end_date,
            max_cloud_cover=request.max_cloud_cover,
            product_type=request.product_type
        )

        return {
            "status": "success",
            "downloaded_files": len(result),
            "output_dir": "data/raw/sentinel2"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unpack")
async def unpack_zip_files():
    """
    Descomprime archivos ZIP descargados.
    """
    try:
        from data.unpack_sentinel2 import unpack_sentinel2_zips

        extracted = unpack_sentinel2_zips("data/raw/sentinel2")

        return {
            "status": "success",
            "extracted_folders": len(extracted),
            "folders": extracted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === ENDPOINTS DE PROCESAMIENTO ===
@app.post("/compute")
async def compute_indices_endpoint(request: ProcessRequest):
    """
    Calcula índices espectrales para todas las carpetas procesadas.
    Genera JSON con etiquetas de estrés vegetal.
    """
    try:
        from data.compute_indices import process_dataset

        output_json = process_dataset(request.raw_dir, request.output_json)

        # Leer y devolver resultados
        with open(output_json, 'r') as f:
            dataset = json.load(f)

        return {
            "status": "success",
            "samples_processed": len(dataset),
            "output_json": output_json,
            "data": dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset")
async def get_dataset():
    """
    Devuelve el dataset procesado completo.
    """
    dataset_path = "data/processed/dataset_labels.json"
    if not Path(dataset_path).exists():
        raise HTTPException(status_code=404, detail="Dataset no encontrado. Ejecuta /compute primero.")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    return dataset


# === ENDPOINTS DE GENERACIÓN DE IMÁGENES ===
@app.post("/generate-images")
async def generate_images_endpoint(request: GenerateRequest):
    """
    Genera imágenes de entrada para modelo LFM2-VL (falso color NIR-Red-Green).
    """
    try:
        from data.generate_images import generate_images_from_dataset

        generated = generate_images_from_dataset(
            raw_dir=request.raw_dir,
            output_dir=request.output_dir,
            dataset_json=request.dataset_json,
            size=request.image_size
        )

        return {
            "status": "success",
            "images_generated": len(generated),
            "output_dir": request.output_dir,
            "images": generated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images")
async def list_images():
    """
    Lista todas las imágenes generadas.
    """
    images_dir = Path("data/images")
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail="No hay imágenes generadas.")

    images = list(images_dir.glob("*.png"))
    return {
        "count": len(images),
        "images": [str(img) for img in images]
    }


# === ENDPOINT DE PIPELINE COMPLETO ===
@app.post("/pipeline")
async def run_full_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Ejecuta el flujo completo: descarga → descompresión → procesamiento → imágenes.

    Retorna inmediatamente y ejecuta en segundo plano.
    """
    def run_pipeline():
        try:
            print("=== Iniciando pipeline Sentinel-2 ===\n")

            # 1. Descargar
            print("[1/4] Descargando imágenes...")
            from data.download_sentinel2 import download_sentinel2
            download_sentinel2(
                footprint=request.footprint,
                start_date=request.start_date,
                end_date=request.end_date,
                max_cloud_cover=request.max_cloud_cover
            )

            # 2. Descomprimir
            print("\n[2/4] Descomprimiendo archivos...")
            from data.unpack_sentinel2 import unpack_sentinel2_zips
            unpack_sentinel2_zips("data/raw/sentinel2")

            # 3. Calcular índices
            print("\n[3/4] Calculando índices espectrales...")
            from data.compute_indices import process_dataset
            process_dataset("data/raw/sentinel2", "data/processed/dataset_labels.json")

            # 4. Generar imágenes
            print("\n[4/4] Generando imágenes para modelo...")
            from data.generate_images import generate_images_from_dataset
            generate_images_from_dataset(
                raw_dir="data/raw/sentinel2",
                output_dir="data/images",
                dataset_json="data/processed/dataset_labels.json"
            )

            print("\n=== Pipeline completado exitosamente ===")

        except Exception as e:
            print(f"\nERROR en pipeline: {e}")
            import traceback
            traceback.print_exc()

    # Ejecutar en segundo plano
    background_tasks.add_task(run_pipeline)

    return {
        "status": "started",
        "message": "Pipeline ejecutándose en segundo plano",
        "parameters": request.dict()
    }


# === ENDPOINTS DE DATOS ===
@app.get("/stats")
async def get_statistics():
    """
    Devuelve estadísticas del dataset procesado.
    """
    dataset_path = Path("data/processed/dataset_labels.json")
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Calcular estadísticas
    total = len(dataset)
    labels = {}
    ndvi_values = []

    for item in dataset:
        label = item['label']
        labels[label] = labels.get(label, 0) + 1
        ndvi_values.append(item['indices']['ndvi_mean'])

    return {
        "total_samples": total,
        "label_distribution": labels,
        "ndvi_stats": {
            "mean": float(np.mean(ndvi_values)) if ndvi_values else None,
            "std": float(np.std(ndvi_values)) if ndvi_values else None,
            "min": float(min(ndvi_values)) if ndvi_values else None,
            "max": float(max(ndvi_values)) if ndvi_values else None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
