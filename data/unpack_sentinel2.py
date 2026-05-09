"""
Descompresión de archivos ZIP de Sentinel-2.
Extrae cada ZIP a su propia carpeta dentro de data/raw/sentinel2.
"""

import zipfile
from pathlib import Path


def unpack_sentinel2_zips(zip_dir: str, output_dir: str = None) -> list:
    """
    Descomprime todos los archivos ZIP de Sentinel-2.

    Args:
        zip_dir: Directorio con archivos ZIP (.zip)
        output_dir: Directorio de extracción (por defecto, mismo directorio)

    Returns:
        Lista de carpetas extraídas
    """
    zip_path = Path(zip_dir)
    extracted = []

    # Buscar archivos ZIP
    zip_files = list(zip_path.glob("*.zip"))

    if not zip_files:
        print(f"No se encontraron archivos ZIP en {zip_dir}")
        return extracted

    print(f"Descomprimiendo {len(zip_files)} archivos ZIP...")

    for zip_file in zip_files:
        try:
            # Determinar directorio de extracción
            if output_dir:
                extract_to = Path(output_dir) / zip_file.stem
            else:
                extract_to = zip_file.parent / zip_file.stem

            extract_to.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(extract_to)

            extracted.append(str(extract_to))
            print(f"  ✓ {zip_file.name} → {extract_to}")

        except zipfile.BadZipFile:
            print(f"  ✗ {zip_file.name}: archivo ZIP corrupto")
        except Exception as e:
            print(f"  ✗ {zip_file.name}: {e}")

    print(f"Descomprimidos {len(extracted)}/{len(zip_files)} archivos")
    return extracted


if __name__ == '__main__':
    import sys

    zip_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sentinel2"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    unpack_sentinel2_zips(zip_dir, output_dir)
