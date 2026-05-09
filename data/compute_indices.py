"""
Cálculo de índices espectrales de Sentinel-2 para detección de estrés vegetal.
Bandas necesarias: B04 (rojo), B08 (NIR), B11 (SWIR1), B12 (SWIR2).
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_sentinel_bands(folder_path: str) -> Dict[str, np.ndarray]:
    """
    Carga bandas B04, B08, B11, B12 desde una carpeta Sentinel-2 descomprimida.

    Args:
        folder_path: Ruta a carpeta con archivos JP2 de Sentinel-2

    Returns:
        Diccionario con arrays numpy para cada banda
    """
    bands = {}
    folder = Path(folder_path)

    # Mapeo de nombres de archivo Sentinel-2 a bandas
    band_patterns = {
        'B04': '*B04_10m.jp2',
        'B08': '*B08_10m.jp2',
        'B11': '*B11_20m.jp2',
        'B12': '*B12_20m.jp2',
    }

    for band_name, pattern in band_patterns.items():
        files = list(folder.rglob(pattern))
        if files:
            with rasterio.open(files[0]) as src:
                bands[band_name] = src.read(1).astype(np.float32)
        else:
            print(f"Advertencia: No se encontró {band_name} en {folder_path}")

    # Validar bandas mínimas
    if 'B04' not in bands or 'B08' not in bands:
        raise ValueError(f"Faltan bandas B04 y/o B08 en {folder_path}")

    return bands


def compute_indices(bands: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calcula índices de vegetación y estrés hídrico.

    Índices calculados:
    - NDVI: Normalized Difference Vegetation Index (salud vegetal)
    - NDWI: Normalized Difference Water Index (estrés hídrico)
    - SWIR_RATIO: Ratio SWIR (daño foliar)

    Returns:
        Diccionario con estadísticas (media, desviación estándar)
    """
    b04 = bands['B04']
    b08 = bands['B08']
    b11 = bands.get('B11', b08)  # fallback a B08 si no hay SWIR
    b12 = bands.get('B12', b08)

    eps = 1e-10

    # NDVI: -1 a 1, sano = 0.6-0.9, estrés = 0.2-0.5, muy malo < 0.2
    ndvi = (b08 - b04) / (b08 + b04 + eps)

    # NDWI: -1 a 1, estrés hídrico cuando < -0.1
    ndwi = (b08 - b11) / (b08 + b11 + eps)

    # SWIR ratio: daño foliar activo cuando > 2.0
    swir_ratio = b11 / (b12 + eps)

    return {
        'ndvi_mean': float(np.nanmean(ndvi)),
        'ndvi_std': float(np.nanstd(ndvi)),
        'ndwi_mean': float(np.nanmean(ndwi)),
        'ndwi_std': float(np.nanstd(ndwi)),
        'swir_ratio_mean': float(np.nanmean(swir_ratio)),
        'swir_ratio_std': float(np.nanstd(swir_ratio)),
        'ndvi_map': ndvi,  # array 2D para visualización
    }


def classify_stress(indices: Dict[str, float]) -> Tuple[str, int]:
    """
    Clasifica el nivel de estrés basado en índices espectrales.

    Reglas:
    - healthy: NDVI >= 0.5 y NDWI > -0.1
    - mild_stress: NDVI >= 0.3 O (NDVI >= 0.5 y NDWI <= -0.2)
    - severe_stress: NDVI < 0.3

    Returns:
        Tupla (etiqueta, id_etiqueta)
    """
    ndvi = indices['ndvi_mean']
    ndwi = indices['ndwi_mean']

    if ndvi >= 0.5 and ndwi > -0.1:
        return 'healthy', 0
    elif ndvi >= 0.3 or (ndvi >= 0.5 and ndwi <= -0.2):
        return 'mild_stress', 1
    else:
        return 'severe_stress', 2


def process_folder(folder_path: str) -> Optional[Dict]:
    """
    Procesa una carpeta completa de Sentinel-2.

    Args:
        folder_path: Ruta a carpeta descomprimida

    Returns:
        Diccionario con resultados o None si hay error
    """
    try:
        bands = load_sentinel_bands(folder_path)
        indices = compute_indices(bands)
        label, label_id = classify_stress(indices)

        return {
            'folder': str(folder_path),
            'filename': Path(folder_path).name,
            'indices': {k: v for k, v in indices.items() if k != 'ndvi_map'},
            'label': label,
            'label_id': label_id,
        }
    except Exception as e:
        print(f'Error en {folder_path}: {e}')
        return None


def process_dataset(raw_dir: str, output_json: str) -> str:
    """
    Procesa todas las carpetas de Sentinel-2 en un directorio.

    Args:
        raw_dir: Directorio con carpetas descomprimidas
        output_json: Ruta donde guardar el JSON con etiquetas

    Returns:
        Ruta del archivo JSON generado
    """
    dataset = []
    raw_path = Path(raw_dir)

    # Buscar carpetas (no archivos zip)
    folders = [f for f in raw_path.iterdir() if f.is_dir()]

    print(f"Procesando {len(folders)} carpetas...")

    for folder in folders:
        result = process_folder(str(folder))
        if result:
            dataset.append(result)

    # Guardar JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        import json
        json.dump(dataset, f, indent=2)

    print(f'Dataset generado: {len(dataset)} muestras en {output_json}')
    return output_json


if __name__ == '__main__':
    import sys

    # Parámetros por defecto
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sentinel2"
    output_json = sys.argv[2] if len(sys.argv) > 2 else "data/processed/dataset_labels.json"

    process_dataset(raw_dir, output_json)
