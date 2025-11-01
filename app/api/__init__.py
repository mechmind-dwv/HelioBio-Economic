"""
🌐 API Routes Module - HelioBio-Economic
Autor: Benjamin Cabeza Durán (mechmind-dwv)
Asistente: DeepSeek AI

Configuración principal de routers y endpoints de la API
"""

from fastapi import APIRouter
from app.api.endpoints import solar, economic, correlation, prediction, system

# Router principal
api_router = APIRouter()

# Incluir todos los routers de endpoints
api_router.include_router(solar.router, prefix="/solar", tags=["Solar Data"])
api_router.include_router(economic.router, prefix="/economic", tags=["Economic Data"])
api_router.include_router(correlation.router, prefix="/correlation", tags=["Correlation Analysis"])
api_router.include_router(prediction.router, prefix="/prediction", tags=["Predictions"])
api_router.include_router(system.router, prefix="/system", tags=["System"])
