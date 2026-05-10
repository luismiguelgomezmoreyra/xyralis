"""
Descarga de imágenes Sentinel-2 desde Copernicus Data Space Ecosystem (CDSE).
Reemplaza al antiguo SciHub (sentinelsat).
"""

import os
import sys
import requests
import json
import time
from pathlib import Path
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

load_dotenv()

def get_token():
    """Obtiene token de acceso para CDSE."""
    username = os.getenv("COPERNICUS_USER")
    password = os.getenv("COPERNICUS_PASSWORD")
    
    if not username or not password:
        print("ERROR: Define COPERNICUS_USER y COPERNICUS_PASSWORD en .env")
        sys.exit(1)

    # Para cuentas personales en CDSE, se usa el client_id 'cdse-public'
    data = {
        'client_id': 'cdse-public',
        'grant_type': 'password',
        'username': username,
        'password': password
    }
    
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()['access_token']
    except Exception as e:
        print(f"Error obteniendo token: {e}")
        if response.status_code == 401:
            print("Error 401: Credenciales inválidas o cuenta no verificada en CDSE.")
        sys.exit(1)

def download_sentinel2(
    footprint: str = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "data/raw/sentinel2",
    max_cloud_cover: int = 50,
    product_type: str = "S2MSI2A"
):
    """
    Descarga imágenes Sentinel-2 desde CDSE usando OData API.
    """
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if footprint is None:
        # Valle del Mantaro aproximado (Punto central)
        footprint = "POINT(-75.2 -12.05)"
    
    # CDSE usa OData. Filtro por punto y tipo MSIL2A
    filter_query = (
        "ContentDate/Start ge 2024-06-01T00:00:00.000Z "
        "and ContentDate/Start le 2024-06-30T23:59:59.999Z "
        "and Collection/Name eq 'SENTINEL-2' "
        "and contains(Name, 'MSIL2A') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
    )
    
    catalogue_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    params = {
        "$filter": filter_query,
        "$top": 1 # Solo 1 para la prueba inicial rápida
    }
    
    print(f"Buscando productos en CDSE...")
    try:
        response = requests.get(catalogue_url, params=params)
        response.raise_for_status()
        products = response.json().get('value', [])
    except Exception as e:
        print(f"Error en la búsqueda: {e}")
        return []

    print(f"Productos encontrados: {len(products)}")
    
    if not products:
        print("No se encontraron imágenes. Prueba ampliando el rango o área.")
        return []

    downloaded_files = []
    for product in products:
        p_id = product['Id']
        p_name = product['Name']
        print(f"Descargando {p_name} ({p_id})...")
        
        download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({p_id})/$value"
        
        # Stream download
        dest_path = Path(output_dir) / f"{p_name}.zip"
        if dest_path.exists():
            print(f"Archivo ya existe: {p_name}")
            downloaded_files.append(dest_path)
            continue
            
        try:
            with requests.get(download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Completado: {p_name}")
            downloaded_files.append(dest_path)
        except Exception as e:
            print(f"Error descargando {p_name}: {e}")
            
    return downloaded_files

if __name__ == "__main__":
    download_sentinel2()
