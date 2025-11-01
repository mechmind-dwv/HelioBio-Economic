"""
🌞 Solar Data Endpoints - HelioBio-Economic
Endpoints para datos y análisis solares de NASA DONKI
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.api.models.responses import (
    StandardResponse, 
    SolarActivityResponse,
    SolarHistoricalQuery
)
from app.services.nasa_solar_service import nasa_solar_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/current", response_model=SolarActivityResponse)
async def get_current_solar_activity():
    """
    Obtener actividad solar actual consolidada
    
    Returns:
        Actividad solar actual con datos de NASA DONKI, NOAA SWPC y SILSO
    """
    try:
        logger.info("🌞 Solicitando actividad solar actual")
        
        activity_data = await nasa_solar_service.get_current_solar_activity()
        
        return SolarActivityResponse(
            success=True,
            message="Actividad solar actual obtenida correctamente",
            data={
                "solar_activity": {
                    "timestamp": activity_data.timestamp.isoformat(),
                    "sunspot_number": activity_data.sunspot_number,
                    "solar_flux": activity_data.solar_flux,
                    "kp_index": activity_data.kp_index,
                    "wind_speed": activity_data.wind_speed,
                    "proton_flux": activity_data.proton_flux,
                    "electron_flux": activity_data.electron_flux,
                    "xray_flux": activity_data.xray_flux,
                    "geomagnetic_field": activity_data.geomagnetic_field
                },
                "data_sources": ["NASA DONKI", "NOAA SWPC", "SILSO"],
                "update_frequency": "60 seconds"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo actividad solar: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo actividad solar: {str(e)}"
        )

@router.get("/historical", response_model=SolarActivityResponse)
async def get_historical_solar_data(
    years: int = Query(50, description="Años de datos históricos", ge=1, le=100),
    include_cycles: bool = Query(True, description="Incluir ciclos solares identificados")
):
    """
    Obtener datos solares históricos
    
    Args:
        years: Número de años de datos históricos (1-100)
        include_cycles: Incluir ciclos solares identificados
    
    Returns:
        Datos solares históricos y análisis de ciclos
    """
    try:
        logger.info(f"📊 Solicitando datos solares históricos ({years} años)")
        
        historical_data = await nasa_solar_service.get_historical_solar_data(years)
        
        response_data = {
            "period_years": years,
            "data_points": len(historical_data),
            "date_range": {
                "start": historical_data.index[0].isoformat() if not historical_data.empty else None,
                "end": historical_data.index[-1].isoformat() if not historical_data.empty else None
            },
            "historical_data": historical_data.reset_index().to_dict('records')
        }
        
        if include_cycles and not historical_data.empty:
            solar_cycles = nasa_solar_service.identify_solar_cycles(historical_data)
            response_data["solar_cycles"] = [
                {
                    "cycle_number": i + 1,
                    "start_date": cycle.start_date.isoformat(),
                    "end_date": cycle.end_date.isoformat(),
                    "peak_date": cycle.peak_date.isoformat(),
                    "period_years": cycle.period_years,
                    "amplitude": cycle.amplitude,
                    "confidence": cycle.confidence
                }
                for i, cycle in enumerate(solar_cycles)
            ]
        
        return SolarActivityResponse(
            success=True,
            message=f"Datos solares históricos de {years} años obtenidos correctamente",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo datos históricos solares: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo datos históricos solares: {str(e)}"
        )

@router.get("/flares")
async def get_solar_flares(days: int = Query(7, description="Días hacia atrás", ge=1, le=30)):
    """
    Obtener fulguraciones solares recientes
    
    Args:
        days: Número de días hacia atrás para buscar (1-30)
    
    Returns:
        Lista de fulguraciones solares recientes
    """
    try:
        logger.info(f"🔥 Solicitando fulguraciones solares ({days} días)")
        
        flares = await nasa_solar_service.get_solar_flares(days)
        
        return SolarActivityResponse(
            success=True,
            message=f"Fulguraciones solares de los últimos {days} días obtenidas",
            data={
                "period_days": days,
                "total_flares": len(flares),
                "flares": [
                    {
                        "flare_id": flare.flare_id,
                        "class_type": flare.class_type,
                        "peak_time": flare.peak_time.isoformat(),
                        "duration_minutes": flare.duration_minutes,
                        "active_region": flare.active_region,
                        "intensity": flare.intensity
                    }
                    for flare in flares
                ],
                "summary": {
                    "by_class": {
                        "X": len([f for f in flares if f.class_type == 'X']),
                        "M": len([f for f in flares if f.class_type == 'M']),
                        "C": len([f for f in flares if f.class_type == 'C']),
                        "B": len([f for f in flares if f.class_type == 'B']),
                        "A": len([f for f in flares if f.class_type == 'A'])
                    }
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo fulguraciones solares: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo fulguraciones solares: {str(e)}"
        )

@router.get("/cme")
async def get_coronal_mass_ejections(days: int = Query(7, description="Días hacia atrás", ge=1, le=30)):
    """
    Obtener Eyecciones de Masa Coronal recientes
    
    Args:
        days: Número de días hacia atrás para buscar (1-30)
    
    Returns:
        Lista de CMEs recientes
    """
    try:
        logger.info(f"🌪️ Solicitando CMEs ({days} días)")
        
        cmes = await nasa_solar_service.get_coronal_mass_ejections(days)
        
        return SolarActivityResponse(
            success=True,
            message=f"CMEs de los últimos {days} días obtenidas",
            data={
                "period_days": days,
                "total_cmes": len(cmes),
                "cmes": [
                    {
                        "cme_id": cme.cme_id,
                        "start_time": cme.start_time.isoformat(),
                        "speed_km_s": cme.speed_km_s,
                        "angle_degrees": cme.angle_degrees,
                        "half_angle": cme.half_angle,
                        "catalog": cme.catalog
                    }
                    for cme in cmes
                ],
                "summary": {
                    "average_speed": np.mean([cme.speed_km_s for cme in cmes]) if cmes else 0,
                    "fastest_cme": max([cme.speed_km_s for cme in cmes]) if cmes else 0
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo CMEs: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo CMEs: {str(e)}"
        )

@router.get("/forecast")
async def get_solar_forecast():
    """
    Obtener pronóstico de actividad solar
    
    Returns:
        Pronóstico de actividad solar a 27 días
    """
    try:
        logger.info("🔮 Solicitando pronóstico solar")
        
        forecast = await nasa_solar_service.get_solar_forecast()
        
        return SolarActivityResponse(
            success=True,
            message="Pronóstico solar obtenido correctamente",
            data={
                "forecast": forecast,
                "issued_at": datetime.now().isoformat(),
                "valid_until": (datetime.now() + timedelta(days=27)).isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo pronóstico solar: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo pronóstico solar: {str(e)}"
        )
