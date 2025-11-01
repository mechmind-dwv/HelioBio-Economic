"""
💹 economic_data_service.py
Servicio de Datos Económicos y Financieros
Autor: Benjamin Cabeza Durán (mechmind-dwv)
Asistente: DeepSeek AI

Servicio unificado para obtener datos económicos de múltiples fuentes:
- Yahoo Finance (mercados bursátiles en tiempo real)
- FRED API (indicadores macroeconómicos de la Reserva Federal)
- World Bank API (datos económicos globales)
- Alpha Vantage (datos financieros avanzados)
"""

import logging
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import os
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

class MarketIndex(Enum):
    """Principales índices bursátiles"""
    SP500 = "^GSPC"
    DOW_JONES = "^DJI"
    NASDAQ = "^IXIC"
    RUSSELL_2000 = "^RUT"
    FTSE_100 = "^FTSE"
    DAX = "^GDAXI"
    NIKKEI_225 = "^N225"

class EconomicIndicator(Enum):
    """Indicadores macroeconómicos clave"""
    GDP = "GDP"                          # Producto Interno Bruto
    CPI = "CPIAUCSL"                     # Índice de Precios al Consumidor
    UNEMPLOYMENT = "UNRATE"              # Tasa de Desempleo
    INTEREST_RATE = "FEDFUNDS"           # Tasa de Fondos Federales
    MONEY_SUPPLY = "M2SL"                # Oferta Monetaria M2
    INDUSTRIAL_PRODUCTION = "INDPRO"     # Producción Industrial
    CONSUMER_SENTIMENT = "UMCSENT"       # Sentimiento del Consumidor
    HOUSING_STARTS = "HOUST"             # Inicios de Vivienda

@dataclass
class MarketData:
    """Datos de mercado bursátil"""
    symbol: str
    timestamp: datetime
    price: float
    change: float
    change_percent: float
    volume: int
    open_price: float
    high: float
    low: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None

@dataclass
class EconomicIndicatorData:
    """Datos de indicador económico"""
    indicator: str
    timestamp: datetime
    value: float
    unit: str
    frequency: str
    previous_value: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None

@dataclass
class EconomicOutlook:
    """Perspectiva económica consolidada"""
    timestamp: datetime
    growth_outlook: str  # "strong", "moderate", "weak"
    inflation_pressure: str  # "high", "moderate", "low"
    employment_health: str  # "strong", "stable", "weak"
    market_sentiment: str  # "bullish", "neutral", "bearish"
    risk_assessment: str  # "low", "medium", "high"
    key_risks: List[str]
    opportunities: List[str]

class EconomicDataService:
    """
    Servicio unificado para datos económicos y financieros
    Integra múltiples fuentes en una interfaz coherente
    """
    
    def __init__(self):
        # Configuración de APIs
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        self.world_bank_key = os.getenv('WORLD_BANK_KEY')
        
        # Clientes de APIs
        self.fred_client = Fred(api_key=self.fred_api_key) if self.fred_api_key else None
        self.session = None
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Configuración de símbolos e indicadores
        self.market_indices = {
            'SP500': '^GSPC',
            'DOW_JONES': '^DJI', 
            'NASDAQ': '^IXIC',
            'RUSSELL_2000': '^RUT',
            'FTSE_100': '^FTSE',
            'DAX': '^GDAXI',
            'NIKKEI_225': '^N225'
        }
        
        self.economic_indicators = {
            'GDP': 'GDP',
            'INFLATION': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE',
            'INTEREST_RATE': 'FEDFUNDS',
            'MONEY_SUPPLY': 'M2SL',
            'INDUSTRIAL_PRODUCTION': 'INDPRO',
            'CONSUMER_SENTIMENT': 'UMCSENT',
            'HOUSING_STARTS': 'HOUST',
            'RETAIL_SALES': 'RSAFS',
            'DURABLE_GOODS': 'DGORDER'
        }
        
        logger.info("💹 Inicializado Servicio de Datos Económicos")
    
    async def initialize(self):
        """Inicializar sesión HTTP asíncrona"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("✅ Sesión HTTP Económica inicializada")
    
    async def close(self):
        """Cerrar sesión HTTP"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("🔒 Sesión HTTP Económica cerrada")
    
    async def check_health(self) -> Dict[str, Any]:
        """Verificar estado de todas las fuentes de datos económicos"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "yahoo_finance": await self._check_yahoo_health(),
            "fred_api": await self._check_fred_health(),
            "alpha_vantage": await self._check_alpha_vantage_health(),
            "world_bank": await self._check_world_bank_health(),
            "overall_status": "healthy"
        }
        
        # Determinar estado general
        services = [health_status['yahoo_finance'], health_status['fred_api']]
        if all(s['status'] == 'healthy' for s in services):
            health_status['overall_status'] = 'healthy'
        elif any(s['status'] == 'unhealthy' for s in services):
            health_status['overall_status'] = 'unhealthy'
        else:
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    async def _check_yahoo_health(self) -> Dict[str, Any]:
        """Verificar estado de Yahoo Finance"""
        try:
            # Probar con un símbolo conocido
            sp500 = yf.Ticker("^GSPC")
            info = sp500.info
            return {
                "status": "healthy",
                "response_time": "fast",
                "last_success": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Yahoo Finance health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_success": None
            }
    
    async def _check_fred_health(self) -> Dict[str, Any]:
        """Verificar estado de FRED API"""
        if not self.fred_client:
            return {
                "status": "unhealthy",
                "error": "FRED API key not configured",
                "last_success": None
            }
        
        try:
            # Probar con un indicador conocido
            test_data = self.fred_client.get_series('GDP', limit=1)
            return {
                "status": "healthy" if not test_data.empty else "degraded",
                "response_time": "fast",
                "last_success": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"FRED API health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_success": None
            }
    
    async def _check_alpha_vantage_health(self) -> Dict[str, Any]:
        """Verificar estado de Alpha Vantage"""
        if not self.alpha_vantage_key:
            return {
                "status": "unconfigured",
                "error": "Alpha Vantage API key not configured",
                "last_success": None
            }
        
        try:
            # Probar conexión básica
            test_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={self.alpha_vantage_key}"
            async with self.session.get(test_url) as response:
                status = response.status
                return {
                    "status": "healthy" if status == 200 else "degraded",
                    "response_time": "fast",
                    "last_success": datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning(f"Alpha Vantage health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_success": None
            }
    
    async def _check_world_bank_health(self) -> Dict[str, Any]:
        """Verificar estado de World Bank API"""
        try:
            test_url = "http://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.CD?format=json&per_page=1"
            async with self.session.get(test_url) as response:
                status = response.status
                return {
                    "status": "healthy" if status == 200 else "degraded",
                    "response_time": "fast",
                    "last_success": datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning(f"World Bank API health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_success": None
            }
    
    async def get_market_data(self, symbol: str = "^GSPC", period: str = "1y") -> Dict[str, Any]:
        """
        Obtener datos de mercado para un símbolo específico
        
        Args:
            symbol: Símbolo del instrumento (ej: ^GSPC, AAPL, etc.)
            period: Período de datos ('1d', '5d', '1mo', '1y', '10y')
            
        Returns:
            Datos de mercado completos
        """
        cache_key = f"market_{symbol}_{period}"
        if cache_key in self.cache and datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_duration:
            return self.cache[cache_key]['data']
        
        try:
            logger.info(f"📊 Obteniendo datos de mercado para {symbol}")
            
            # Usar yfinance para datos de mercado
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            info_data = ticker.info
            
            # Procesar datos históricos
            market_data = []
            for date, row in hist_data.iterrows():
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=date.to_pydatetime(),
                    price=row['Close'],
                    change=row['Close'] - row['Open'],
                    change_percent=((row['Close'] - row['Open']) / row['Open']) * 100,
                    volume=row['Volume'],
                    open_price=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    market_cap=info_data.get('marketCap'),
                    pe_ratio=info_data.get('trailingPE')
                ))
            
            result = {
                "symbol": symbol,
                "period": period,
                "data_points": len(market_data),
                "current_price": market_data[-1].price if market_data else 0,
                "market_data": [md.__dict__ for md in market_data],
                "metadata": {
                    "company_name": info_data.get('longName', ''),
                    "sector": info_data.get('sector', ''),
                    "industry": info_data.get('industry', ''),
                    "currency": info_data.get('currency', 'USD')
                }
            }
            
            # Cachear resultados
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': result
            }
            
            logger.info(f"✅ Obtenidos {len(market_data)} puntos de datos para {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos de mercado para {symbol}: {e}")
            return {
                "symbol": symbol,
                "period": period,
                "data_points": 0,
                "current_price": 0,
                "market_data": [],
                "metadata": {},
                "error": str(e)
            }
    
    async def get_economic_indicators(self) -> Dict[str, Any]:
        """
        Obtener indicadores macroeconómicos actuales
        
        Returns:
            Diccionario con todos los indicadores económicos clave
        """
        cache_key = "economic_indicators"
        if cache_key in self.cache and datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(hours=1):
            return self.cache[cache_key]['data']
        
        try:
            logger.info("📈 Obteniendo indicadores económicos...")
            
            indicators_data = {}
            
            # Obtener cada indicador de FRED
            for indicator_name, fred_code in self.economic_indicators.items():
                try:
                    indicator_data = await self._get_fred_indicator(fred_code, indicator_name)
                    if indicator_data:
                        indicators_data[indicator_name] = indicator_data
                except Exception as e:
                    logger.warning(f"Error obteniendo indicador {indicator_name}: {e}")
                    continue
            
            # Calcular métricas derivadas
            economic_health = self._assess_economic_health(indicators_data)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "indicators": indicators_data,
                "economic_health": economic_health,
                "summary": self._generate_economic_summary(indicators_data)
            }
            
            # Cachear resultados (los indicadores económicos cambian lentamente)
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': result
            }
            
            logger.info(f"✅ Obtenidos {len(indicators_data)} indicadores económicos")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo indicadores económicos: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "indicators": {},
                "economic_health": {},
                "summary": {},
                "error": str(e)
            }
    
    async def _get_fred_indicator(self, fred_code: str, indicator_name: str) -> Optional[EconomicIndicatorData]:
        """Obtener un indicador específico de FRED"""
        if not self.fred_client:
            return None
        
        try:
            series_data = self.fred_client.get_series(fred_code, limit=10)
            if series_data.empty:
                return None
            
            # Tomar los dos últimos valores para calcular cambios
            current_value = series_data.iloc[-1]
            previous_value = series_data.iloc[-2] if len(series_data) > 1 else None
            
            change = current_value - previous_value if previous_value else None
            change_percent = (change / previous_value * 100) if previous_value and previous_value != 0 else None
            
            return EconomicIndicatorData(
                indicator=indicator_name,
                timestamp=series_data.index[-1].to_pydatetime(),
                value=current_value,
                unit=self._get_indicator_unit(indicator_name),
                frequency="Monthly",  # FRED principalmente mensual
                previous_value=previous_value,
                change=change,
                change_percent=change_percent
            )
            
        except Exception as e:
            logger.warning(f"Error obteniendo indicador FRED {fred_code}: {e}")
            return None
    
    def _get_indicator_unit(self, indicator_name: str) -> str:
        """Obtener unidad de medida para un indicador"""
        units = {
            'GDP': 'Billions of Dollars',
            'INFLATION': 'Index',
            'UNEMPLOYMENT': 'Percent',
            'INTEREST_RATE': 'Percent',
            'MONEY_SUPPLY': 'Billions of Dollars',
            'INDUSTRIAL_PRODUCTION': 'Index',
            'CONSUMER_SENTIMENT': 'Index',
            'HOUSING_STARTS': 'Thousands of Units',
            'RETAIL_SALES': 'Millions of Dollars',
            'DURABLE_GOODS': 'Millions of Dollars'
        }
        return units.get(indicator_name, 'Unknown')
    
    def _assess_economic_health(self, indicators: Dict[str, EconomicIndicatorData]) -> Dict[str, str]:
        """Evaluar salud económica basada en indicadores"""
        health_assessment = {}
        
        # Evaluar crecimiento (GDP)
        if 'GDP' in indicators:
            gdp_growth = indicators['GDP'].change_percent or 0
            if gdp_growth > 3:
                health_assessment['growth'] = 'strong'
            elif gdp_growth > 1:
                health_assessment['growth'] = 'moderate'
            else:
                health_assessment['growth'] = 'weak'
        
        # Evaluar inflación
        if 'INFLATION' in indicators:
            inflation = indicators['INFLATION'].change_percent or 0
            if inflation > 5:
                health_assessment['inflation'] = 'high'
            elif inflation > 2:
                health_assessment['inflation'] = 'moderate'
            else:
                health_assessment['inflation'] = 'low'
        
        # Evaluar empleo
        if 'UNEMPLOYMENT' in indicators:
            unemployment = indicators['UNEMPLOYMENT'].value or 0
            if unemployment < 4:
                health_assessment['employment'] = 'strong'
            elif unemployment < 6:
                health_assessment['employment'] = 'stable'
            else:
                health_assessment['employment'] = 'weak'
        
        # Evaluar sentimiento
        if 'CONSUMER_SENTIMENT' in indicators:
            sentiment = indicators['CONSUMER_SENTIMENT'].value or 0
            if sentiment > 90:
                health_assessment['sentiment'] = 'bullish'
            elif sentiment > 70:
                health_assessment['sentiment'] = 'neutral'
            else:
                health_assessment['sentiment'] = 'bearish'
        
        return health_assessment
    
    def _generate_economic_summary(self, indicators: Dict[str, EconomicIndicatorData]) -> Dict[str, Any]:
        """Generar resumen económico ejecutivo"""
        summary = {
            "key_highlights": [],
            "areas_of_concern": [],
            "outlook": "stable",
            "momentum": "neutral"
        }
        
        # Análisis de GDP
        if 'GDP' in indicators and indicators['GDP'].change_percent:
            gdp_growth = indicators['GDP'].change_percent
            if gdp_growth > 3:
                summary["key_highlights"].append(f"Fuerte crecimiento del PIB: {gdp_growth:.1f}%")
                summary["momentum"] = "positive"
            elif gdp_growth < 1:
                summary["areas_of_concern"].append(f"Crecimiento débil del PIB: {gdp_growth:.1f}%")
                summary["momentum"] = "negative"
        
        # Análisis de inflación
        if 'INFLATION' in indicators and indicators['INFLATION'].change_percent:
            inflation = indicators['INFLATION'].change_percent
            if inflation > 5:
                summary["areas_of_concern"].append(f"Alta inflación: {inflation:.1f}%")
                summary["outlook"] = "cautious"
            elif inflation < 1:
                summary["key_highlights"].append(f"Baja inflación: {inflation:.1f}%")
        
        # Análisis de empleo
        if 'UNEMPLOYMENT' in indicators:
            unemployment = indicators['UNEMPLOYMENT'].value
            if unemployment < 4:
                summary["key_highlights"].append(f"Bajo desempleo: {unemployment:.1f}%")
            elif unemployment > 7:
                summary["areas_of_concern"].append(f"Alto desempleo: {unemployment:.1f}%")
        
        return summary
    
    async def get_market_conditions(self) -> Dict[str, Any]:
        """
        Obtener condiciones actuales del mercado
        
        Returns:
            Análisis de condiciones de mercado
        """
        try:
            # Obtener datos de múltiples índices
            indices_data = {}
            volatility_measures = []
            
            for index_name, symbol in list(self.market_indices.items())[:3]:  # Primeros 3 índices
                try:
                    market_data = await self.get_market_data(symbol, "1mo")
                    if market_data and 'market_data' in market_data:
                        prices = [item['price'] for item in market_data['market_data']]
                        if len(prices) > 1:
                            volatility = np.std(prices) / np.mean(prices) * 100
                            volatility_measures.append(volatility)
                        
                        current_price = market_data['current_price']
                        indices_data[index_name] = {
                            'price': current_price,
                            'volatility': volatility if 'volatility' in locals() else 0,
                            'trend': 'up' if prices[-1] > prices[0] else 'down'
                        }
                except Exception as e:
                    logger.warning(f"Error obteniendo índice {index_name}: {e}")
                    continue
            
            # Calcular volatilidad promedio
            avg_volatility = np.mean(volatility_measures) if volatility_measures else 0
            
            # Determinar condición del mercado
            market_condition = self._determine_market_condition(indices_data, avg_volatility)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "indices": indices_data,
                "volatility_index": avg_volatility,
                "market_condition": market_condition,
                "recommendations": self._generate_market_recommendations(market_condition)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo condiciones de mercado: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "indices": {},
                "volatility_index": 0,
                "market_condition": "unknown",
                "error": str(e)
            }
    
    def _determine_market_condition(self, indices_data: Dict, volatility: float) -> str:
        """Determinar condición general del mercado"""
        if not indices_data:
            return "unknown"
        
        # Contar tendencias alcistas vs bajistas
        up_count = sum(1 for data in indices_data.values() if data.get('trend') == 'up')
        down_count = sum(1 for data in indices_data.values() if data.get('trend') == 'down')
        
        # Evaluar volatilidad
        if volatility > 3:
            volatility_status = "high"
        elif volatility > 1.5:
            volatility_status = "moderate"
        else:
            volatility_status = "low"
        
        # Determinar condición general
        if up_count > down_count and volatility_status == "low":
            return "bullish"
        elif up_count > down_count and volatility_status == "high":
            return "volatile_bull"
        elif down_count > up_count and volatility_status == "high":
            return "bearish"
        elif down_count > up_count and volatility_status == "low":
            return "correction"
        else:
            return "neutral"
    
    def _generate_market_recommendations(self, market_condition: str) -> List[str]:
        """Generar recomendaciones basadas en condición de mercado"""
        recommendations = {
            "bullish": [
                "Considerar exposición a crecimiento",
                "Mantener diversificación",
                "Evaluar toma de ganancias en posiciones sobrevaloradas"
            ],
            "volatile_bull": [
                "Enfoque en calidad de activos",
                "Mantener liquidez para oportunidades",
                "Considerar coberturas de volatilidad"
            ],
            "bearish": [
                "Enfoque defensivo en cartera",
                "Aumentar asignación a activos refugio",
                "Considerar estrategias de corto plazo"
            ],
            "correction": [
                "Oportunidad para compra de calidad a precios bajos",
                "Revisar stops de protección",
                "Mantener perspectiva de largo plazo"
            ],
            "neutral": [
                "Mantener estrategia actual",
                "Revisar rebalanceo de cartera",
                "Monitorear indicadores económicos clave"
            ],
            "unknown": [
                "Recolectar más datos de mercado",
                "Consultar múltiples fuentes",
                "Mantener posición conservadora"
            ]
        }
        
        return recommendations.get(market_condition, [])
    
    async def get_long_term_economic_data(self) -> pd.DataFrame:
        """
        Obtener datos económicos de largo plazo para análisis de ciclos
        
        Returns:
            DataFrame con datos económicos históricos
        """
        cache_key = "long_term_economic"
        if cache_key in self.cache and datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(hours=6):
            return self.cache[cache_key]['data']
        
        try:
            logger.info("📊 Obteniendo datos económicos de largo plazo...")
            
            # Obtener datos históricos de múltiples indicadores
            economic_series = {}
            
            if self.fred_client:
                # GDP histórico (trimestral)
                gdp_data = self.fred_client.get_series('GDP', observation_start='1970-01-01')
                if not gdp_data.empty:
                    economic_series['GDP'] = gdp_data
                
                # Inflación histórica (mensual)
                cpi_data = self.fred_client.get_series('CPIAUCSL', observation_start='1970-01-01')
                if not cpi_data.empty:
                    economic_series['INFLATION'] = cpi_data
                
                # Desempleo histórico (mensual)
                unemp_data = self.fred_client.get_series('UNRATE', observation_start='1970-01-01')
                if not unemp_data.empty:
                    economic_series['UNEMPLOYMENT'] = unemp_data
            
            # Combinar series en un DataFrame
            economic_df = pd.DataFrame()
            for name, series in economic_series.items():
                if economic_df.empty:
                    economic_df = pd.DataFrame({name: series})
                else:
                    economic_df[name] = series
            
            # Rellenar valores faltantes y calcular métricas derivadas
            economic_df = economic_df.ffill().bfill()
            economic_df['GDP_GROWTH'] = economic_df['GDP'].pct_change() * 100 if 'GDP' in economic_df else 0
            
            # Cachear resultados
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': economic_df
            }
            
            logger.info(f"✅ Obtenidos {len(economic_df)} registros históricos económicos")
            return economic_df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos económicos de largo plazo: {e}")
            # Retornar datos de muestra en caso de error
            return self._generate_sample_economic_data()
    
    def _generate_sample_economic_data(self) -> pd.DataFrame:
        """Generar datos económicos de muestra para desarrollo"""
        dates = pd.date_range(start='1970-01-01', end=datetime.now(), freq='Q')
        
        # Simular datos económicos con ciclos
        np.random.seed(42)  # Para reproducibilidad
        
        # GDP con tendencia de crecimiento y ciclos
        trend = np.linspace(1000, 20000, len(dates))
        cycle = 5000 * np.sin(2 * np.pi * np.arange(len(dates)) / 40)  # Ciclo ~10 años
        noise = np.random.normal(0, 500, len(dates))
        gdp = trend + cycle + noise
        
        # Inflación con ciclos
        inflation_cycle = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 20)  # Ciclo ~5 años
        inflation = 2 + inflation_cycle + np.random.normal(0, 1, len(dates))
        
        # Desempleo contra-cíclico
        unemployment_cycle = -2 * np.sin(2 * np.pi * np.arange(len(dates)) / 40)  # Opuesto a GDP
        unemployment = 6 + unemployment_cycle + np.random.normal(0, 0.5, len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'GDP': gdp,
            'INFLATION': inflation,
            'UNEMPLOYMENT': unemployment,
            'GDP_GROWTH': np.gradient(gdp) / gdp * 100
        }).set_index('date')
    
    async def get_economic_outlook(self) -> EconomicOutlook:
        """
        Generar perspectiva económica basada en datos actuales
        
        Returns:
            Perspectiva económica consolidada
        """
        try:
            # Obtener datos actuales
            indicators = await self.get_economic_indicators()
            market_conditions = await self.get_market_conditions()
            
            # Evaluar outlook basado en múltiples factores
            health = indicators.get('economic_health', {})
            
            outlook = EconomicOutlook(
                timestamp=datetime.now(),
                growth_outlook=health.get('growth', 'moderate'),
                inflation_pressure=health.get('inflation', 'moderate'),
                employment_health=health.get('employment', 'stable'),
                market_sentiment=market_conditions.get('market_condition', 'neutral'),
                risk_assessment=self._assess_overall_risk(health, market_conditions),
                key_risks=self._identify_key_risks(health),
                opportunities=self._identify_opportunities(health, market_conditions)
            )
            
            return outlook
            
        except Exception as e:
            logger.error(f"Error generando perspectiva económica: {e}")
            # Retornar outlook por defecto
            return EconomicOutlook(
                timestamp=datetime.now(),
                growth_outlook="moderate",
                inflation_pressure="moderate", 
                employment_health="stable",
                market_sentiment="neutral",
                risk_assessment="medium",
                key_risks=["Datos económicos incompletos"],
                opportunities=["Monitorear indicadores clave"]
            )
    
    def _assess_overall_risk(self, health: Dict, market_conditions: Dict) -> str:
        """Evaluar riesgo económico general"""
        risk_factors = 0
        
        if health.get('growth') == 'weak':
            risk_factors += 2
        elif health.get('growth') == 'strong':
            risk_factors -= 1
            
        if health.get('inflation') == 'high':
            risk_factors += 2
            
        if health.get('employment') == 'weak':
            risk_factors += 2
            
        if market_conditions.get('market_condition') in ['bearish', 'volatile_bull']:
            risk_factors += 1
        
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"
    
    def _identify_key_risks(self, health: Dict) -> List[str]:
        """Identificar riesgos económicos clave"""
        risks = []
        
        if health.get('growth') == 'weak':
            risks.append("Crecimiento económico débil")
            
        if health.get('inflation') == 'high':
            risks.append("Presiones inflacionarias")
            
        if health.get('employment') == 'weak':
            risks.append("Mercado laboral débil")
            
        if health.get('sentiment') == 'bearish':
            risks.append("Sentimiento del consumidor negativo")
        
        if not risks:
            risks.append("Riesgos económicos moderados")
            
        return risks
    
    def _identify_opportunities(self, health: Dict, market_conditions: Dict) -> List[str]:
        """Identificar oportunidades económicas"""
        opportunities = []
        
        if health.get('growth') == 'strong':
            opportunities.append("Entorno de crecimiento favorable")
            
        if health.get('inflation') == 'low':
            opportunities.append("Estabilidad de precios")
            
        if health.get('employment') == 'strong':
            opportunities.append("Mercado laboral robusto")
            
        if market_conditions.get('market_condition') == 'bullish':
            opportunities.append("Mercados alcistas")
        elif market_conditions.get('market_condition') == 'correction':
            opportunities.append("Oportunidades de compra en corrección")
        
        if not opportunities:
            opportunities.append("Estabilidad económica general")
            
        return opportunities

# Instancia global para uso en otros módulos
economic_data_service = EconomicDataService()
