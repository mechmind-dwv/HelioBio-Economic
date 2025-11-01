"""
🔗 Correlation Analysis Endpoints - HelioBio-Economic
Endpoints para análisis de correlación solar-económica
"""

import logging
from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from app.api.models.responses import (
    StandardResponse, 
    CorrelationResponse,
    CorrelationQuery
)
from app.services.nasa_solar_service import nasa_solar_service
from app.services.economic_data_service import economic_data_service
from app.services.correlation_service import correlation_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/solar-economic", response_model=CorrelationResponse)
async def get_solar_economic_correlation(
    economic_indicator: str = Query("SP500", description="Indicador económico"),
    solar_indicator: str = Query("sunspots", description="Indicador solar"),
    period_years: int = Query(50, description="Período de análisis en años", ge=10, le=100)
):
    """
    Análisis de correlación entre indicadores solares y económicos
    
    Args:
        economic_indicator: Indicador económico (SP500, GDP, etc.)
        solar_indicator: Indicador solar (sunspots, solar_flux, etc.)
        period_years: Período de análisis en años
    
    Returns:
        Análisis completo de correlación con múltiples métodos
    """
    try:
        logger.info(f"🔗 Analizando correlación {economic_indicator}-{solar_indicator}")
        
        # Obtener datos históricos
        economic_data = await economic_data_service.get_long_term_economic_data()
        solar_data = await nasa_solar_service.get_historical_solar_data(period_years)
        
        # Preparar series específicas
        if economic_indicator in economic_data.columns:
            economic_series = economic_data[economic_indicator]
        else:
            # Usar SP500 como default
            economic_series = await economic_data_service.get_market_data("^GSPC", f"{period_years}y")
            economic_series = pd.Series(
                [item['price'] for item in economic_series['market_data']],
                index=pd.to_datetime([item['timestamp'] for item in economic_series['market_data']])
            )
        
        if solar_indicator in solar_data.columns:
            solar_series = solar_data[solar_indicator]
        else:
            # Usar manchas solares como default
            solar_series = solar_data['sunspots']
        
        # Realizar análisis de correlación
        correlation_result = await correlation_service.analyze_correlation(
            economic_series, solar_series, economic_indicator, solar_indicator
        )
        
        return CorrelationResponse(
            success=True,
            message=f"Correlación {economic_indicator}-{solar_indicator} analizada",
            data={
                "analysis_parameters": {
                    "economic_indicator": economic_indicator,
                    "solar_indicator": solar_indicator,
                    "period_years": period_years,
                    "sample_size": correlation_result.sample_size
                },
                "correlation_results": {
                    "methods": correlation_result.methods,
                    "optimal_lag_months": correlation_result.optimal_lag,
                    "lag_correlation": correlation_result.lag_correlation,
                    "p_value": correlation_result.p_value,
                    "confidence_interval": correlation_result.confidence_interval,
                    "significance": correlation_result.significance.value
                },
                "statistical_tests": {
                    "stationarity": correlation_result.stationarity_test
                },
                "interpretation": {
                    "notes": correlation_result.notes,
                    "strength": correlation_result.significance.value,
                    "direction": "Positive" if correlation_result.methods['pearson'] > 0 else "Negative"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de correlación: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en análisis de correlación: {str(e)}"
        )

@router.get("/cross-spectral")
async def get_cross_spectral_analysis():
    """
    Análisis espectral cruzado entre ciclos solares y económicos
    
    Returns:
        Análisis de frecuencias comunes y sincronización
    """
    try:
        logger.info("📊 Realizando análisis espectral cruzado")
        
        # Obtener datos para análisis espectral
        economic_data = await economic_data_service.get_long_term_economic_data()
        solar_data = await nasa_solar_service.get_historical_solar_data(50)
        
        # Usar SP500 y manchas solares para análisis
        economic_series = await economic_data_service.get_market_data("^GSPC", "50y")
        economic_series = pd.Series(
            [item['price'] for item in economic_series['market_data']],
            index=pd.to_datetime([item['timestamp'] for item in economic_series['market_data']])
        )
        
        solar_series = solar_data['sunspots']
        
        # Realizar análisis espectral
        spectral_analysis = await correlation_service.cross_spectral_analysis(
            economic_series, solar_series
        )
        
        return CorrelationResponse(
            success=True,
            message="Análisis espectral cruzado completado",
            data={
                "spectral_analysis": {
                    "common_periods": spectral_analysis.common_periods,
                    "coherence_spectrum": spectral_analysis.coherence_spectrum,
                    "phase_synchronization": spectral_analysis.phase_synchronization,
                    "shared_cycles": spectral_analysis.shared_cycles,
                    "dominant_frequencies": spectral_analysis.dominant_frequencies
                },
                "wavelet_analysis": {
                    "max_coherence": spectral_analysis.wavelet_coherence.get('max_coherence', 0),
                    "high_coherence_regions": "detected" if spectral_analysis.wavelet_coherence.get('high_coherence_regions')[0].size > 0 else "none"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error en análisis espectral: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en análisis espectral: {str(e)}"
        )

@router.get("/causality")
async def analyze_causality():
    """
    Análisis de causalidad entre variables solares y económicas
    
    Returns:
        Tests de causalidad de Granger y entropía de transferencia
    """
    try:
        logger.info("🔍 Analizando causalidad solar-económica")
        
        # Obtener datos
        economic_data = await economic_data_service.get_long_term_economic_data()
        solar_data = await nasa_solar_service.get_historical_solar_data(30)
        
        economic_series = economic_data['GDP_GROWTH'] if 'GDP_GROWTH' in economic_data.columns else economic_data.iloc[:, 0]
        solar_series = solar_data['sunspots']
        
        # Realizar análisis de causalidad
        causality_analysis = await correlation_service.analyze_causality(
            economic_series, solar_series
        )
        
        return CorrelationResponse(
            success=True,
            message="Análisis de causalidad completado",
            data={
                "causality_analysis": {
                    "cause": causality_analysis.cause,
                    "effect": causality_analysis.effect,
                    "direction": causality_analysis.direction,
                    "confidence": causality_analysis.confidence,
                    "granger_causality": causality_analysis.granger_causality,
                    "transfer_entropy": causality_analysis.transfer_entropy,
                    "convergent_cross_mapping": causality_analysis.convergent_cross_mapping
                },
                "interpretation": {
                    "relationship_strength": "Strong" if causality_analysis.confidence > 0.7 else "Moderate" if causality_analysis.confidence > 0.5 else "Weak",
                    "implied_relationship": f"{causality_analysis.cause} → {causality_analysis.effect}"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de causalidad: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en análisis de causalidad: {str(e)}"
        )

@router.get("/common-cycles")
async def get_common_cycles():
    """
    Encontrar ciclos comunes entre dominios solar y económico
    
    Returns:
        Ciclos comunes identificados y análisis de sincronización
    """
    try:
        logger.info("🔄 Buscando ciclos comunes solar-económicos")
        
        common_cycles = correlation_service.find_common_cycles()
        
        return CorrelationResponse(
            success=True,
            message="Ciclos comunes identificados",
            data=common_cycles
        )
        
    except Exception as e:
        logger.error(f"❌ Error buscando ciclos comunes: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error buscando ciclos comunes: {str(e)}"
        )
