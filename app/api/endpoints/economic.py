"""
💹 Economic Data Endpoints - HelioBio-Economic
Endpoints para datos económicos y financieros
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.api.models.responses import (
    StandardResponse, 
    EconomicDataResponse,
    MarketDataQuery
)
from app.services.economic_data_service import economic_data_service
from app.core.kondratiev_analysis import kondratiev_analyzer

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/markets", response_model=EconomicDataResponse)
async def get_market_data(
    symbol: str = Query("^GSPC", description="Símbolo del mercado (SP500, ^DJI, etc)"),
    period: str = Query("1y", description="Período (1d, 5d, 1mo, 1y, 10y)")
):
    """
    Obtener datos de mercados bursátiles
    
    Args:
        symbol: Símbolo del instrumento financiero
        period: Período de datos históricos
    
    Returns:
        Datos de mercado históricos y actuales
    """
    try:
        logger.info(f"📈 Solicitando datos de mercado: {symbol} ({period})")
        
        market_data = await economic_data_service.get_market_data(symbol, period)
        
        return EconomicDataResponse(
            success=True,
            message=f"Datos de mercado para {symbol} obtenidos correctamente",
            data=market_data
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo datos de mercado: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo datos de mercado: {str(e)}"
        )

@router.get("/indicators", response_model=EconomicDataResponse)
async def get_economic_indicators():
    """
    Obtener indicadores macroeconómicos principales
    
    Returns:
        Indicadores económicos actuales de FRED y otras fuentes
    """
    try:
        logger.info("📊 Solicitando indicadores económicos")
        
        indicators = await economic_data_service.get_economic_indicators()
        
        return EconomicDataResponse(
            success=True,
            message="Indicadores económicos obtenidos correctamente",
            data=indicators
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo indicadores económicos: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo indicadores económicos: {str(e)}"
        )

@router.get("/kondratiev", response_model=EconomicDataResponse)
async def get_kondratiev_analysis():
    """
    Análisis de ondas largas de Kondratiev
    
    Returns:
        Análisis completo de la onda Kondratiev actual y predicciones
    """
    try:
        logger.info("🌊 Solicitando análisis Kondratiev")
        
        analysis = kondratiev_analyzer.analyze_long_waves()
        report = kondratiev_analyzer.generate_kondratiev_report()
        
        return EconomicDataResponse(
            success=True,
            message="Análisis Kondratiev completado",
            data={
                "current_analysis": {
                    "current_wave": analysis.current_wave.wave_number,
                    "current_phase": analysis.current_phase.value,
                    "phase_progress": analysis.phase_progress,
                    "next_phase_transition": analysis.next_phase_transition.isoformat()
                },
                "solar_synchronization": analysis.solar_correlation,
                "economic_implications": analysis.economic_implications,
                "risk_assessment": analysis.risk_assessment,
                "full_report": report
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error en análisis Kondratiev: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en análisis Kondratiev: {str(e)}"
        )

@router.get("/conditions")
async def get_market_conditions():
    """
    Obtener condiciones actuales del mercado
    
    Returns:
        Análisis de condiciones de mercado y volatilidad
    """
    try:
        logger.info("📊 Solicitando condiciones de mercado")
        
        conditions = await economic_data_service.get_market_conditions()
        
        return EconomicDataResponse(
            success=True,
            message="Condiciones de mercado analizadas",
            data=conditions
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo condiciones de mercado: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo condiciones de mercado: {str(e)}"
        )

@router.get("/outlook")
async def get_economic_outlook():
    """
    Obtener perspectiva económica consolidada
    
    Returns:
        Perspectiva económica basada en múltiples indicadores
    """
    try:
        logger.info("🔮 Solicitando perspectiva económica")
        
        outlook = await economic_data_service.get_economic_outlook()
        
        return EconomicDataResponse(
            success=True,
            message="Perspectiva económica generada",
            data={
                "outlook": {
                    "growth_outlook": outlook.growth_outlook,
                    "inflation_pressure": outlook.inflation_pressure,
                    "employment_health": outlook.employment_health,
                    "market_sentiment": outlook.market_sentiment,
                    "risk_assessment": outlook.risk_assessment,
                    "key_risks": outlook.key_risks,
                    "opportunities": outlook.opportunities
                },
                "timestamp": outlook.timestamp.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error generando perspectiva económica: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error generando perspectiva económica: {str(e)}"
        )
