# Xyralis

Sistema de procesamiento de imágenes Sentinel-2 para detección de estrés agrícola.

## 📋 Características

- **Descarga directa** desde Copernicus Open Access Hub
- **Cálculo de índices espectrales**: NDVI, NDWI, SWIR Ratio
- **Clasificación automática** de estrés vegetal (saludable, estrés leve, estrés severo)
- **Generación de imágenes** en falso color NIR-Red-Green para modelos LFM2-VL
- **API REST** para orquestar el flujo completo

## 🗂️ Estructura del proyecto

```
xyralis/
├── main.py                  # API FastAPI
├── data/
│   ├── download_sentinel2.py    # Descarga desde Copernicus
│   ├── unpack_sentinel2.py      # Descompresión de ZIP
│   ├── compute_indices.py       # Cálculo de índices espectrales
│   ├── generate_images.py       # Generación de imágenes para modelo
│   ├── raw/
│   │   └── sentinel2/           # ZIP y carpetas descomprimidas
│   ├── processed/
│   │   └── dataset_labels.json  # Dataset con etiquetas
│   └── images/                  # Imágenes PNG para LFM2-VL (224x224)
├── .env.example            # Variables de entorno de ejemplo
└── requirements.txt         # Dependencias Python
```

## 🚀 Instalación

```bash
# 1. Clonar repositorio
cd /mnt/datos/dev/2026/Mayo/space/xyralis

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\\Scripts\\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

## 🔐 Configuración de credenciales Copernicus

1. Regístrate en [Copernicus Open Access Hub](https://register.copernicus.eu)
2. Copia `.env.example` a `.env`:
```bash
cp .env.example .env
```
3. Edita `.env` con tus credenciales:
```env
COPERNICUS_USER=tu_usuario
COPERNICUS_PASSWORD=tu_password
```

## 📡 Uso de la API

Iniciar servidor:
```bash
python main.py
# o
uvicorn main:app --reload
```

API disponible en `http://localhost:8000`

### Endpoints principales

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Estado del servicio |
| `/health` | GET | Verificación de dependencias |
| `/download` | POST | Descargar imágenes Sentinel-2 |
| `/unpack` | POST | Descomprimir archivos ZIP |
| `/compute` | POST | Calcular índices espectrales |
| `/generate-images` | POST | Generar imágenes para modelo |
| `/pipeline` | POST | Ejecutar flujo completo |
| `/dataset` | GET | Obtener dataset etiquetado |
| `/images` | GET | Listar imágenes generadas |
| `/stats` | GET | Estadísticas del dataset |

### Ejemplo: Pipeline completo

```bash
curl -X POST "http://localhost:8000/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "footprint": "POLYGON((-75.5 -12.2, -74.8 -12.2, -74.8 -11.8, -75.5 -11.8, -75.5 -12.2))",
    "start_date": "20240601",
    "end_date": "20241001",
    "max_cloud_cover": 20
  }'
```

## 🧮 Índices espectrales calculados

| Índice | Fórmula | Rango | Interpretación |
|--------|---------|-------|----------------|
| NDVI | (NIR - Red) / (NIR + Red) | -1 a 1 | >0.5: saludable, 0.3-0.5: estrés leve, <0.3: estrés severo |
| NDWI | (NIR - SWIR1) / (NIR + SWIR1) | -1 a 1 | >-0.1: normal, <-0.1: estrés hídrico |
| SWIR Ratio | SWIR1 / SWIR2 | >0 | >2.0: daño foliar activo |

## 🖼️ Formato de salida para LFM2-VL

Las imágenes generadas en `data/images/` tienen:
- **Formato**: PNG
- **Tamaño**: 224×224 píxeles
- **Canales**: 3 (RGB en falso color NIR-Red-Green)
- **Normalización**: Percentiles 2-98 para eliminar outliers
- **Nombrado**: `{identificador}_{etiqueta}.png` (ej: `S2A_20240615_healthy.png`)

### Composición de color falso

| Canal | Banda Sentinel-2 | Aplicación |
|-------|------------------|------------|
| Rojo | B08 (NIR) | Vegetación sana aparece brillante |
| Verde | B04 (Red) | Contraste de vigor |
| Azul | B03 (Green) | Suelo y contexto |

## 📊 Clasificación de estrés

El sistema asigna etiquetas automáticas basadas en umbrales de NDVI y NDWI:

| Etiqueta | ID | Condición |
|----------|----|-----------|
| `healthy` | 0 | NDVI ≥ 0.5 y NDWI > -0.1 |
| `mild_stress` | 1 | NDVI ≥ 0.3 O (NDVI ≥ 0.5 y NDWI ≤ -0.2) |
| `severe_stress` | 2 | NDVI < 0.3 |

## 🔧 Ejecución manual (CLI)

```bash
# 1. Descargar imágenes
python data/download_sentinel2.py

# 2. Descomprimir ZIP
python data/unpack_sentinel2.py

# 3. Calcular índices y etiquetas
python data/compute_indices.py

# 4. Generar imágenes para modelo
python data/generate_images.py
```

## 📁 Estructura de datos generada

```
data/
├── raw/
│   └── sentinel2/
│       ├── S2A_MSIL2A_*.zip          # Archivos descargados
│       └── S2A_MSIL2A_*/             # Carpetas descomprimidas
│           ├── *_B03_10m.jp2
│           ├── *_B04_10m.jp2
│           ├── *_B08_10m.jp2
│           └── *_B11_20m.jp2
├── processed/
│   └── dataset_labels.json           # Dataset JSON con índices y etiquetas
└── images/
    ├── S2A_20240601_healthy.png
    ├── S2A_20240601_mild_stress.png
    └── S2A_20240601_severe_stress.png
```

### Formato JSON del dataset

```json
[
  {
    "folder": "data/raw/sentinel2/S2A_MSIL2A_...",
    "filename": "S2A_MSIL2A_...",
    "indices": {
      "ndvi_mean": 0.72,
      "ndvi_std": 0.08,
      "ndwi_mean": -0.05,
      "ndwi_std": 0.03,
      "swir_ratio_mean": 1.45,
      "swir_ratio_std": 0.12
    },
    "label": "healthy",
    "label_id": 0
  }
]
```

## ⚙️ Parámetros configurables

| Parámetro | Ubicación | Descripción |
|-----------|-----------|-------------|
| `footprint` | API `/download` o `/pipeline` | Polígono WKT del área de interés |
| `start_date` / `end_date` | API | Rango de fechas (YYYYMMDD) |
| `max_cloud_cover` | API | Porcentaje máximo de nubes (0-100) |
| `product_type` | Código | `S2MSI2A` (L2A, corregido atmosféricamente) |
| `image_size` | API `/generate-images` | Tamaño de imagen en píxeles (ancho, alto) |

## 🧪 Testing del sistema

```bash
# Verificar instalación
curl http://localhost:8000/health

# Probar descarga (pequeña área, 1 día)
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{"start_date":"20240601","end_date":"20240601","max_cloud_cover":10}'

# Ver estadísticas
curl http://localhost:8000/stats
```

## 📚 Referencias

- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- [Sentinel-2 Product Types](https://sentinels.copernicus.eu/web/sentinel/user-guides/document-library/-/asset_publisher/4rwww1rG6mSc/content/sentinel-2-product-types)
- [NDVI interpretation](https://earthobservatory.nasa.gov/features/MeasuringVegetation)
- [sentinelsat Python library](https://sentinelsat.readthedocs.io/)

## 🐛 Troubleshooting

**Error: "COPERNICUS_USER no definido"**
→ Crea archivo `.env` con tus credenciales Copernicus

**Error: "No se encontraron productos"**
→ Ajusta fechas, área o reduce `max_cloud_cover`

**Error: "Faltan bandas B03/B11/B12"**
→ Verifica que la escena descargada incluya todas las bandas requeridas

**PDF2: "ZIP corrupto"**
→ Descarga nuevamente, el archivo puede estar incompleto

## 📄 Licencia

Proyecto académico - Uso educativo y de investigación.
