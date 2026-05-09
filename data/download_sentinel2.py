"""
Descarga de imágenes Sentinel-2 desde Copernicus Open Access Hub.
Requiere credenciales de Copernicus en variables de entorno:
- COPERNICUS_USER
- COPERNICUS_PASSWORD
"""

import os
import sys
from pathlib import Path
from datetime import date
from sentinelsat import SentinelAPI
from dotenv import load_dotenv

load_dotenv()

def download_sentinel2(
    footprint: str = None,
    start_date: str = "20240601",
    end_date: str = "20241001",
    output_dir: str = "data/raw/sentinel2",
    max_cloud_cover: int = 20,
    product_type: str = "S2MSI2A",
    platform: str = "Sentinel-2"
):
    """
    Descarga imágenes Sentinel-2 que cumplan los criterios especificados.

    Args:
        footprint: WKT polygon del área de interés (ej: "POLYGON((-75.5 -12.2...))")
        start_date: Fecha inicio en formato YYYYMMDD
        end_date: Fecha fin en formato YYYYMMDD
        output_dir: Directorio de destino
        max_cloud_cover: Porcentaje máximo de nubes (0-100)
        product_type: Tipo de producto (S2MSI2A = L2A corregido atmosféricamente)
        platform: Plataforma (Sentinel-2)
    """
    # Credenciales
    user = os.getenv("COPERNICUS_USER")
    password = os.getenv("COPERNICUS_PASSWORD")

    if not user or not password:
        print("ERROR: Define COPERNICUS_USER y COPERNICUS_PASSWORD en .env")
        sys.exit(1)

    # Conectar a API
    api = SentinelAPI(
        user,
        password,
        'https://apihub.copernicus.eu/apihub'
    )

    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Footprint por defecto: Valle del Mantaro, Perú
    if footprint is None:
        footprint = "POLYGON((-75.5 -12.2, -74.8 -12.2, -74.8 -11.8, -75.5 -11.8, -75.5 -12.2))"

    print(f"Buscando productos en área: {footprint[:50]}...")
    print(f"Rango fechas: {start_date} a {end_date}")
    print(f"Nubosidad máxima: {max_cloud_cover}%")

    # Buscar productos
    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname=platform,
        cloudcoverpercentage=(0, max_cloud_cover),
        producttype=product_type
    )

    print(f"Productos encontrados: {len(products)}")

    if not products:
        print("No se encontraron imágenes. Ajusta los parámetros de búsqueda.")
        return []

    # Descargar
    print(f"Descargando a {output_dir}...")
    api.download_all(products, directory_path=output_dir)

    downloaded = list(Path(output_dir).glob("*.zip"))
    print(f"Descarga completada: {len(downloaded)} archivos")
    return downloaded


if __name__ == "__main__":
    download_sentinel2()
