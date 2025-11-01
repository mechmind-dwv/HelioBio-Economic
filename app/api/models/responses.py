"""
📊 Modelos de Respuesta API - HelioBio-Economic
Estructuras estandarizadas para respuestas de la API
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

class StandardResponse(BaseModel):
    """Respuesta estándar para todos los endpoints"""
    success: bool = Field(..., description="Indica si la solicitud fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo del resultado")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SolarActivityResponse(StandardResponse):
    """Respuesta específica para actividad solar"""
    data: Optional[Dict[str, Any]] = Field(None, description="Datos de actividad solar")

class EconomicDataResponse(StandardResponse):
    """Respuesta específica para datos económicos"""
    data: Optional[Dict[str, Any]] = Field(None, description="Datos económicos")

class CorrelationResponse(StandardResponse):
    """Respuesta específica para análisis de correlación"""
    data: Optional[Dict[str, Any]] = Field(None, description="Resultados de correlación")

class PredictionResponse(StandardResponse):
    """Respuesta específica para predicciones"""
    data: Optional[Dict[str, Any]] = Field(None, description="Resultados de predicción")

class HealthResponse(StandardResponse):
    """Respuesta específica para estado del sistema"""
    data: Optional[Dict[str, Any]] = Field(None, description="Estado de componentes del sistema")

# Modelos para parámetros de query
class SolarHistoricalQuery(BaseModel):
    years: int = Field(50, ge=1, le=100, description="Años de datos históricos")
    include_cycles: bool = Field(True, description="Incluir ciclos solares identificados")

class MarketDataQuery(BaseModel):
    symbol: str = Field("^GSPC", description="Símbolo del mercado")
    period: str = Field("1y", description="Período de datos")

class CorrelationQuery(BaseModel):
    economic_indicator: str = Field("SP500", description="Indicador económico")
    solar_indicator: str = Field("sunspots", description="Indicador solar")
    period_years: int = Field(50, ge=10, le=100, description="Período de análisis")

class PredictionQuery(BaseModel):
    cycles: str = Field("solar,kondratiev", description="Ciclos a incluir")
    horizon_days: int = Field(30, ge=1, le=365, description="Horizonte de predicción")
