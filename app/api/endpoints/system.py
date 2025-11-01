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
        Estado detallado de todos los componentes
    """
    try:
        from app.core.solar_economic_ml import solar_economic_ml
        from app.core.kondratiev_analysis import kondratiev_analyzer
        
        ml_status = "Trained" if solar_economic_ml.is_trained else "Not Trained"
        kondratiev_status = "Analyzed" if kondratiev_analyzer.current_analysis else "Pending"
        
        return HealthResponse(
            success=True,
            message="Estado del sistema obtenido",
            data={
                "core_components": {
                    "solar_economic_ml": {
                        "status": ml_status,
                        "models_trained": len(solar_economic_ml.models) if solar_economic_ml.is_trained else 0,
                        "best_model": max(solar_economic_ml.model_performance.items(), 
                                        key=lambda x: x[1].r2_score)[0] if solar_economic_ml.model_performance else "None"
                    },
                    "kondratiev_analyzer": {
                        "status": kondratiev_status,
                        "current_wave": kondratiev_analyzer.current_analysis.current_wave.wave_number if kondratiev_analyzer.current_analysis else None,
                        "current_phase": kondratiev_analyzer.current_analysis.current_phase.value if kondratiev_analyzer.current_analysis else None
                    }
                },
                "api_endpoints": {
                    "total_endpoints": 20,  # Actualizar con conteo real
                    "active_endpoints": [
                        "/api/solar/current",
                        "/api/economic/indicators", 
                        "/api/correlation/solar-economic",
                        "/api/prediction/economic",
                        "/api/system/health"
                    ]
                },
                "data_sources": {
                    "nasa_donki": "Active",
                    "yahoo_finance": "Active", 
                    "fred_api": "Active" if economic_data_service.fred_client else "Inactive",
                    "noaa_swpc": "Active"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado del sistema: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo estado del sistema: {str(e)}"
        )
