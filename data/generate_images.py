"""
Generación de imágenes de entrada para modelo LFM2-VL a partir de Sentinel-2.
Crea composiciones en falso color NIR-Red-Green que resaltan el estrés vegetal.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import json


def load_sentinel_bands(folder_path: str) -> Dict[str, np.ndarray]:
    """Carga bandas desde carpeta Sentinel-2 (reutiliza lógica de compute_indices)."""
    bands = {}
    folder = Path(folder_path)

    band_patterns = {
        'B04': '*B04_10m.jp2',
        'B08': '*B08_10m.jp2',
        'B03': '*B03_10m.jp2',  # Verde para el canal green
    }

    for band_name, pattern in band_patterns.items():
        files = list(folder.rglob(pattern))
        if files:
            with rasterio.open(files[0]) as src:
                bands[band_name] = src.read(1).astype(np.float32)

    if 'B04' not in bands or 'B08' not in bands:
        raise ValueError(f"Faltan bandas requeridas en {folder_path}")

    # Si no hay B03, usar B04 como sustituto
    if 'B03' not in bands:
        bands['B03'] = bands['B04'].copy()

    return bands


def make_false_color_rgb(
    bands: Dict[str, np.ndarray],
    output_path: str,
    size: tuple = (224, 224)
) -> str:
    """
    Crea composición RGB en falso color NIR-Red-Green.

    Canales:
    - R: B08 (NIR) → vegetación sana = rojo brillante
    - G: B04 (Red)
    - B: B03 (Green)

    La normalización por percentiles elimina outliers (nubes, sombras).
    """
    b08 = bands['B08']
    b04 = bands['B04']
    b03 = bands['B03']

    # Stack en orden RGB
    rgb = np.stack([b08, b04, b03], axis=-1)

    # Normalización robusta por percentiles
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    # Crear imagen y redimensionar
    img = Image.fromarray(rgb).resize(size, Image.LANCZOS)

    # Crear directorio si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    return output_path


def generate_images_from_dataset(
    raw_dir: str,
    output_dir: str,
    dataset_json: Optional[str] = None,
    size: tuple = (224, 224)
) -> list:
    """
    Genera imágenes para todas las muestras Sentinel-2.

    Args:
        raw_dir: Directorio con carpetas descomprimidas
        output_dir: Directorio de salida para imágenes
        dataset_json: Opcional, JSON con metadatos para nombrar archivos
        size: Tupla (ancho, alto) para redimensionar

    Returns:
        Lista de rutas generadas
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargar metadatos si se proporciona
    label_map = {}
    if dataset_json and Path(dataset_json).exists():
        with open(dataset_json, 'r') as f:
            dataset = json.load(f)
            for item in dataset:
                label_map[item['folder']] = item

    folders = [f for f in raw_path.iterdir() if f.is_dir()]
    generated = []

    print(f"Generando imágenes para {len(folders)} carpetas...")

    for folder in folders:
        try:
            bands = load_sentinel_bands(str(folder))

            # Nombre de archivo: si hay etiqueta, incluirla
            if str(folder) in label_map:
                label = label_map[str(folder)]['label']
                filename = f"{folder.name}_{label}.png"
            else:
                filename = f"{folder.name}.png"

            output_file = str(output_path / filename)
            make_false_color_rgb(bands, output_file, size)
            generated.append(output_file)
            print(f"  ✓ {filename}")

        except Exception as e:
            print(f"  ✗ Error en {folder.name}: {e}")

    print(f"Imágenes generadas: {len(generated)} en {output_dir}")
    return generated


if __name__ == '__main__':
    import sys

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sentinel2"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/images"
    dataset_json = sys.argv[3] if len(sys.argv) > 3 else "data/processed/dataset_labels.json"

    generate_images_from_dataset(raw_dir, output_dir, dataset_json)
