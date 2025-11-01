"""
🗃️ Módulo de Datos - HelioBio-Economic
Gestión centralizada de datos solares, económicos y modelos ML
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configuración de rutas de datos
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SOLAR_DATA_DIR = os.path.join(DATA_DIR, 'solar')
ECONOMIC_DATA_DIR = os.path.join(DATA_DIR, 'economic')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

# Crear directorios si no existen
for directory in [DATA_DIR, SOLAR_DATA_DIR, ECONOMIC_DATA_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

logger.info("📁 Estructura de datos inicializada")
