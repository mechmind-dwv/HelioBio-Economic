"""
🎯 Prediction Endpoints - HelioBio-Economic
Endpoints para predicciones económicas basadas en ciclos solares
"""

import logging
from fastapi import APIRouter, HTTPException, Query

from app.api.models.responses import (
    StandardResponse, 
    PredictionResponse,
    PredictionQuery
)
from app.core.solar_economic_ml import solar_economic_ml
from app.services.nasa_solar_service import nasa_solar_service
from app.services.economic_data_service import economic_data_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/economic", response_model=PredictionResponse)
async def predict_economic_cycles(
    cycles: str = Query("solar,kondratiev", description="Ciclos a incluir"),
    horizon_days: int = Query(30, description="Horizonte de predicción en días", ge=1, le=365)
):
    """
    Predecir ciclos económicos basados en factores solares
    
    Args:
        cycles: Lista de ciclos a incluir (solar, kondratiev, kuznets, etc.)
        horizon_days: Horizonte de predicción en días
    
    Returns:
        Predicciones económicas basadas en modelos ML y ciclos solares
    """
    try:
        logger.info(f"🎯 Solicitando predicción económica - Ciclos: {cycles}, Horizonte: {horizon_days}d")
        
        # Entrenar modelos si no están entrenados
        if not solar_economic_ml.is_trained:
            economic_data = await economic_data_service.get_long_term_economic_data()
            solar_data = await nasa_solar_service.get_historical_solar_data(50)
            await solar_economic_ml.train_models(economic_data, solar_data)
        
        # Realizar predicción
        cycle_list = [c.strip() for c in cycles.split(",")]
        predictions = await solar_economic_ml.predict_economic_cycles(cycle_list, horizon_days)
        
        return PredictionResponse(
            success=True,
            message=f"Predicción económica para {horizon_days} días generada",
            data=predictions
        )
        
    except Exception as e:
        logger.error(f"❌ Error en predicción económica: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en predicción económica: {str(e)}"
        )

@router.get("/ml-performance")
async def get_ml_performance():
    """
    Obtener métricas de performance de los modelos ML
    
    Returns:
        Métricas de performance y importancia de características
    """
    try:
        logger.info("🧠 Solicitando métricas de performance ML")
        
        performance = solar_economic_ml.get_model_performance()
        feature_importance = solar_economic_ml.get_feature_importance()
        
        return PredictionResponse(
            success=True,
            message="Métricas de performance ML obtenidas",
            data={
                "model_performance": performance,
                "feature_importance": feature_importance,
                "training_status": solar_economic_ml.is_trained
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo métricas ML: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo métricas ML: {str(e)}"
        )

@router.post("/retrain-models")
async def retrain_ml_models():
    """
    Re-entrenar modelos de Machine Learning
    
    Returns:
        Estado del re-entrenamiento de modelos
    """
    try:
        logger.info("🔄 Re-entrenando modelos ML")
        
        economic_data = await economic_data_service.get_long_term_economic_data()
        solar_data = await nasa_solar_service.get_historical_solar_data(50)
        
        performance = await solar_economic_ml.train_models(economic_data, solar_data)
        
        return PredictionResponse(
            success=True,
            message="Modelos ML re-entrenados correctamente",
            data={
                "training_performance": performance,
                "models_trained": len(performance),
                "best_model": max(performance.items(), key=lambda x: x[1].r2_score)[0] if performance else "None"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error re-entrenando modelos ML: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error re-entrenando modelos ML: {str(e)}"
        )

@router.get("/crisis-risk")
async def predict_crisis_risk():
    """
    Predecir riesgo de crisis basado en ciclos actuales
    
    Returns:
        Evaluación de riesgo de crisis económica
    """
    try:
        logger.info("⚠️ Evaluando riesgo de crisis")
        
        from app.core.economic_cycles import economic_cycle_analyzer
        
        # Obtener ciclos actuales
        economic_data = await economic_data_service.get_long_term_economic_data()
        solar_data = await nasa_solar_service.get_historical_solar_data(50)
        
        economic_cycles = economic_cycle_analyzer.identify_economic_cycles(economic_data)
        solar_cycles = economic_cycle_analyzer.identify_solar_cycles(solar_data)
        
        current_cycles = {
            'economic': economic_cycles,
            'solar': solar_cycles
        }
        
        risk_assessment = economic_cycle_analyzer.predict_next_crisis_risk(current_cycles)
        
        return PredictionResponse(
            success=True,
            message="Evaluación de riesgo de crisis completada",
            data={
                "risk_assessment": risk_assessment,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_sources": ["Economic cycles", "Solar cycles", "Historical patterns"]
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error evaluando riesgo de crisis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error evaluando riesgo de crisis: {str(e)}"
        )
