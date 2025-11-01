"""
⚙️ System Endpoints - HelioBio-Economic
Endpoints para monitoreo y gestión del sistema
"""

import logging
from fastapi import APIRouter
import psutil
import os

from app.api.models.responses import StandardResponse, HealthResponse
from app.services.nasa_solar_service import nasa_solar_service
from app.services.economic_data_service import economic_data_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Verificar estado de salud del sistema completo
    
    Returns:
        Estado de todos los componentes y servicios
    """
    try:
        # Verificar servicios externos
        nasa_health = await nasa_solar_service.check_health()
        economic_health = await economic_data_service.check_health()
        
        # Métricas del sistema
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_memory_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }
        
        # Determinar estado general
        all_healthy = (
            nasa_health.get('overall_status') == 'healthy' and
            economic_health.get('overall_status') == 'healthy' and
            system_metrics['cpu_percent'] < 80 and
            system_metrics['memory_usage'] < 80
        )
        
        return HealthResponse(
            success=True,
            message="Estado del sistema verificado",
            data={
                "system_status": "healthy" if all_healthy else "degraded",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "nasa_solar_service": nasa_health,
                    "economic_data_service": economic_health
                },
                "system_metrics": system_metrics,
                "version": "HelioBio-Economic v1.0.0"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return HealthResponse(
            success=False,
            message="Error verificando estado del sistema",
            error=str(e)
        )

@router.get("/status")
async def system_status():
    """
    Obtener estado detallado del sistema
    
    Returns:
        Estado detallado de
