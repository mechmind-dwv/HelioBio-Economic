Aquí tienes el `main.py` principal para HelioBio-Economic:

```python
#!/usr/bin/env python3
"""
🌞 HelioBio-Economic v1.0
FastAPI Principal - Sistema de Análisis de Correlación Solar-Económica
Autor: Benjamin Cabeza Durán (mechmind-dwv)
Asistente: DeepSeek AI
Email: ia.mechmind@gmail.com

Sistema interdisciplinario que investiga correlaciones entre ciclos solares y económicos
Basado en el legado de Alexander Chizhevsky y Nikolai Kondratiev
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Importaciones internas
from app.core.economic_cycles import EconomicCycleAnalyzer
from app.core.solar_economic_ml import SolarEconomicML
from app.core.kondratiev_analysis import KondratievAnalyzer
from app.services.nasa_solar_service import NASASolarService
from app.services.economic_data_service import EconomicDataService
from app.services.correlation_service import CorrelationService

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/helio_economic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicialización de FastAPI
app = FastAPI(
    title="HelioBio-Economic API",
    description="Sistema de análisis de correlación entre ciclos solares y económicos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servicios globales
nasa_service = NASASolarService()
economic_service = EconomicDataService()
correlation_service = CorrelationService()
economic_analyzer = EconomicCycleAnalyzer()
ml_analyzer = SolarEconomicML()
kondratiev_analyzer = KondratievAnalyzer()

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar la aplicación"""
    logger.info("🚀 Iniciando HelioBio-Economic v1.0")
    
    try:
        # Cargar datos iniciales
        await nasa_service.initialize()
        await economic_service.initialize()
        logger.info("✅ Servicios inicializados correctamente")
        
        # Entrenar modelos ML en segundo plano
        background_tasks = BackgroundTasks()
        background_tasks.add_task(train_ml_models)
        
    except Exception as e:
        logger.error(f"❌ Error en inicialización: {e}")

async def train_ml_models():
    """Entrenar modelos de ML en segundo plano"""
    try:
        logger.info("🧠 Entrenando modelos ML...")
        await ml_analyzer.train_models()
        logger.info("✅ Modelos ML entrenados correctamente")
    except Exception as e:
        logger.error(f"❌ Error entrenando modelos ML: {e}")

# ==================== ENDPOINTS PRINCIPALES ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal del dashboard"""
    with open("app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/api/health")
async def health_check():
    """Verificar estado del sistema"""
    return {
        "status": "healthy",
        "service": "HelioBio-Economic v1.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "nasa_api": await nasa_service.check_health(),
            "economic_data": await economic_service.check_health(),
            "ml_models": ml_analyzer.get_model_status(),
            "correlation_engine": "active"
        }
    }

# ==================== DATOS SOLARES ====================

@app.get("/api/solar/current")
async def get_current_solar_activity():
    """Obtener actividad solar actual"""
    try:
        data = await nasa_service.get_current_solar_activity()
        return {
            "timestamp": datetime.now().isoformat(),
            "solar_activity": data
        }
    except Exception as e:
        logger.error(f"Error obteniendo datos solares: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/solar/historical")
async def get_historical_solar_data(
    years: int = Query(50, description="Años de datos históricos"),
    include_cycles: bool = Query(True, description="Incluir ciclos solares identificados")
):
    """Obtener datos solares históricos"""
    try:
        data = await nasa_service.get_historical_solar_data(years)
        
        if include_cycles:
            cycles = economic_analyzer.identify_solar_cycles(data)
            data["solar_cycles"] = cycles
            
        return data
    except Exception as e:
        logger.error(f"Error obteniendo datos solares históricos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DATOS ECONÓMICOS ====================

@app.get("/api/economic/markets")
async def get_market_data(
    symbol: str = Query("^GSPC", description="Símbolo del mercado (SP500, ^DJI, etc)"),
    period: str = Query("10y", description="Período (1d, 5d, 1mo, 1y, 10y)")
):
    """Obtener datos de mercados bursátiles"""
    try:
        data = await economic_service.get_market_data(symbol, period)
        return {
            "symbol": symbol,
            "period": period,
            "market_data": data
        }
    except Exception as e:
        logger.error(f"Error obteniendo datos de mercado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/economic/indicators")
async def get_economic_indicators():
    """Obtener indicadores macroeconómicos principales"""
    try:
        indicators = await economic_service.get_economic_indicators()
        return {
            "timestamp": datetime.now().isoformat(),
            "economic_indicators": indicators
        }
    except Exception as e:
        logger.error(f"Error obteniendo indicadores económicos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/economic/kondratiev")
async def get_kondratiev_analysis():
    """Análisis de ondas largas de Kondratiev"""
    try:
        analysis = kondratiev_analyzer.analyze_long_waves()
        return {
            "analysis": analysis,
            "current_phase": kondratiev_analyzer.get_current_phase(),
            "next_transition": kondratiev_analyzer.predict_next_transition()
        }
    except Exception as e:
        logger.error(f"Error en análisis Kondratiev: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ANÁLISIS DE CORRELACIÓN ====================

@app.get("/api/correlation/solar-economic")
async def get_solar_economic_correlation(
    economic_indicator: str = Query("SP500", description="Indicador económico"),
    solar_indicator: str = Query("sunspots", description="Indicador solar"),
    period_years: int = Query(50, description="Período de análisis en años")
):
    """Obtener correlación entre indicadores solares y económicos"""
    try:
        correlation_data = await correlation_service.analyze_correlation(
            economic_indicator, solar_indicator, period_years
        )
        
        return {
            "economic_indicator": economic_indicator,
            "solar_indicator": solar_indicator,
            "period_years": period_years,
            "correlation_analysis": correlation_data
        }
    except Exception as e:
        logger.error(f"Error en análisis de correlación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/correlation/cross-spectral")
async def cross_spectral_analysis():
    """Análisis espectral cruzado entre ciclos solares y económicos"""
    try:
        analysis = correlation_service.cross_spectral_analysis()
        return {
            "spectral_analysis": analysis,
            "common_cycles": correlation_service.find_common_cycles()
        }
    except Exception as e:
        logger.error(f"Error en análisis espectral: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PREDICCIÓN Y ML ====================

@app.get("/api/prediction/economic")
async def predict_economic_cycles(
    cycles: str = Query("solar,kondratiev", description="Ciclos a incluir"),
    horizon_days: int = Query(30, description="Horizonte de predicción en días")
):
    """Predecir ciclos económicos basados en factores solares"""
    try:
        cycle_list = [c.strip() for c in cycles.split(",")]
        prediction = await ml_analyzer.predict_economic_cycles(
            cycle_list, horizon_days
        )
        
        return {
            "included_cycles": cycle_list,
            "prediction_horizon_days": horizon_days,
            "predictions": prediction
        }
    except Exception as e:
        logger.error(f"Error en predicción económica: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/retrain")
async def retrain_ml_models(background_tasks: BackgroundTasks):
    """Re-entrenar modelos de ML"""
    background_tasks.add_task(train_ml_models)
    return {"status": "ML model retraining started"}

@app.get("/api/ml/performance")
async def get_ml_performance():
    """Obtener métricas de performance de los modelos ML"""
    try:
        metrics = ml_analyzer.get_model_performance()
        return {
            "model_metrics": metrics,
            "feature_importance": ml_analyzer.get_feature_importance()
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ANÁLISIS DE CICLOS ====================

@app.get("/api/cycles/combined")
async def get_combined_cycle_analysis():
    """Análisis combinado de ciclos solares y económicos"""
    try:
        # Obtener datos solares
        solar_data = await nasa_service.get_historical_solar_data(100)
        solar_cycles = economic_analyzer.identify_solar_cycles(solar_data)
        
        # Obtener datos económicos
        economic_data = await economic_service.get_long_term_economic_data()
        economic_cycles = economic_analyzer.identify_economic_cycles(economic_data)
        
        # Análisis de interferencia
        interference = correlation_service.analyze_cycle_interference(
            solar_cycles, economic_cycles
        )
        
        return {
            "solar_cycles": solar_cycles,
            "economic_cycles": economic_cycles,
            "cycle_interference": interference,
            "synchronization_analysis": correlation_service.analyze_synchronization()
        }
    except Exception as e:
        logger.error(f"Error en análisis combinado de ciclos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cycles/alignment")
async def get_cycle_alignment():
    """Analizar alineación entre ciclos solares y económicos"""
    try:
        alignment = correlation_service.analyze_cycle_alignment()
        return {
            "current_alignment": alignment["current"],
            "historical_alignment": alignment["historical"],
            "predicted_alignment": alignment["predicted"]
        }
    except Exception as e:
        logger.error(f"Error en análisis de alineación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HISTÓRICO Y REPORTES ====================

@app.get("/api/historical/crises")
async def get_historical_crises_analysis():
    """Análisis de crisis históricas en relación con ciclos solares"""
    try:
        crises_analysis = economic_analyzer.analyze_historical_crises()
        return {
            "historical_crises": crises_analysis,
            "solar_correlation": economic_analyzer.calculate_crisis_solar_correlation()
        }
    except Exception as e:
        logger.error(f"Error en análisis de crisis históricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/comprehensive")
async def get_comprehensive_report():
    """Reporte comprensivo de análisis helio-económico"""
    try:
        report = {
            "executive_summary": await generate_executive_summary(),
            "current_analysis": await get_current_analysis(),
            "predictions": await get_predictions_summary(),
            "risk_assessment": await get_risk_assessment()
        }
        return report
    except Exception as e:
        logger.error(f"Error generando reporte comprensivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FUNCIONES AUXILIARES ====================

async def generate_executive_summary():
    """Generar resumen ejecutivo del análisis"""
    return {
        "current_solar_activity": await nasa_service.get_current_solar_activity(),
        "economic_outlook": await economic_service.get_economic_outlook(),
        "cycle_alignment": await get_cycle_alignment(),
        "key_insights": await correlation_service.get_key_insights()
    }

async def get_current_analysis():
    """Obtener análisis actual"""
    return {
        "market_conditions": await economic_service.get_market_conditions(),
        "solar_conditions": await nasa_service.get_solar_conditions(),
        "correlation_strength": await correlation_service.get_current_correlation()
    }

async def get_predictions_summary():
    """Obtener resumen de predicciones"""
    return {
        "economic_forecast": await ml_analyzer.get_economic_forecast(),
        "solar_forecast": await nasa_service.get_solar_forecast(),
        "combined_risk": await ml_analyzer.calculate_combined_risk()
    }

async def get_risk_assessment():
    """Evaluación de riesgo basada en ciclos"""
    return {
        "solar_economic_risk": await correlation_service.assess_risk(),
        "recommendations": await correlation_service.get_recommendations()
    }

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

Y aquí tienes el archivo de requisitos `requirements.txt` correspondiente:

```txt
# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Procesamiento de datos
pandas==2.1.3
numpy==1.25.2
scipy==1.11.3

# Machine Learning
scikit-learn==1.3.2
tensorflow==2.14.0
statsmodels==0.14.0

# APIs y web
requests==2.31.0
aiohttp==3.9.1
yfinance==0.2.18
fredapi==0.5.0
python-dotenv==1.0.0

# Análisis de series temporales
arch==6.2.0
prophet==1.1.4

# Utilidades
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.17.0
jupyter==1.0.0

# Logging y monitoreo
structlog==23.2.0
prometheus-client==0.17.1
```

Este `main.py` incluye:

✅ **FastAPI completo** con todos los endpoints necesarios  
✅ **Gestión de servicios** (NASA, datos económicos, ML)  
✅ **Análisis de correlación** solar-económica  
✅ **Predicciones basadas en ML**  
✅ **Análisis de ciclos** (Kondratiev, solares)  
✅ **Sistema de salud** y monitoreo  
✅ **Logging completo** y manejo de errores  
✅ **Configuración CORS** para desarrollo  
