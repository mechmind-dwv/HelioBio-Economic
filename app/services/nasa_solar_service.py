"""
🌞 nasa_solar_service.py
Servicio de Datos Solares de la NASA DONKI API
Autor: Benjamin Cabeza Durán (mechmind-dwv)
Asistente: DeepSeek AI

Servicio completo para obtener y procesar datos solares en tiempo real de:
- NASA DONKI API (Space Weather Database Of Notifications, Knowledge, Information)
- NOAA Space Weather Prediction Center
- Datos históricos de manchas solares
- Tormentas geomagnéticas y fulguraciones solares
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
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

class SolarEventType(Enum):
    """Tipos de eventos solares"""
    SOLAR_FLARE = "FLR"           # Fulguraciones solares
    CORONAL_MASS_EJECTION = "CME" # Eyecciones de Masa Coronal
    GEOMAGNETIC_STORM = "GST"     # Tormentas geomagnéticas
    SOLAR_ENERGETIC_PARTICLE = "SEP" # Partículas energéticas solares
    MAGNETOPAUSE_CROSSING = "MPC" # Cruces de magnetopausa

@dataclass
class SolarFlare:
    """Estructura para fulguraciones solares"""
    flare_id: str
    class_type: str  # A, B, C, M, X
    peak_time: datetime
    duration_minutes: int
    active_region: str
    intensity: float
    position: Tuple[float, float]  # Lat, Lon solar

@dataclass
class CoronalMassEjection:
    """Estructura para Eyecciones de Masa Coronal"""
    cme_id: str
    start_time: datetime
    speed_km_s: float
    angle_degrees: float
    half_angle: float
    catalog: str
    note: str

@dataclass
class GeomagneticStorm:
    """Estructura para tormentas geomagnéticas"""
    storm_id: str
    start_time: datetime
    kp_index: float
    gst_scale: str  # G1-G5
    estimated_dst: float
    cause: str

@dataclass
class SolarActivitySummary:
    """Resumen de actividad solar"""
    timestamp: datetime
    sunspot_number: int
    solar_flux: float
    kp_index: float
    wind_speed: float
    proton_flux: float
    electron_flux: float
    xray_flux: float
    geomagnetic_field: str

class NASASolarService:
    """
    Servicio para obtener datos solares de NASA DONKI API y otras fuentes
    
    NASA DONKI API Documentation: https://ccmc.gsfc.nasa.gov/tools/DONKI/
    """
    
    def __init__(self):
        self.api_key = os.getenv('NASA_API_KEY', 'DEMO_KEY')
        self.base_url = "https://api.nasa.gov/DONKI"
        self.session = None
        self.cache = {}
        self.cache_duration = timedelta(minutes=10)
        
        # URLs específicas de DONKI
        self.endpoints = {
            'solar_flares': f"{self.base_url}/FLR",
            'coronal_mass_ejections': f"{self.base_url}/CME",
            'geomagnetic_storms': f"{self.base_url}/GST",
            'solar_energetic_particles': f"{self.base_url}/SEP",
            'magnetopause_crossings': f"{self.base_url}/MPC",
            'wsa_enlil_simulations': f"{self.base_url}/WSAEnlilSimulations"
        }
        
        # Fuentes adicionales
        self.noaa_swpc_url = "https://services.swpc.noaa.gov/json"
        self.silso_url = "http://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"
        
        logger.info("🌞 Inicializado Servicio NASA Solar DONKI")
    
    async def initialize(self):
        """Inicializar sesión HTTP asíncrona"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("✅ Sesión HTTP NASA inicializada")
    
    async def close(self):
        """Cerrar sesión HTTP"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("🔒 Sesión HTTP NASA cerrada")
    
    async def check_health(self) -> Dict[str, Any]:
        """Verificar estado del servicio NASA"""
        try:
            # Probar conexión con endpoint simple
            test_url = f"{self.base_url}/FLR?api_key={self.api_key}&startDate=2024-01-01"
            
            async with self.session.get(test_url) as response:
                status = response.status
                return {
                    "status": "healthy" if status == 200 else "degraded",
                    "nasa_api_available": status == 200,
                    "api_key_configured": bool(self.api_key and self.api_key != 'DEMO_KEY'),
                    "last_check": datetime.now().isoformat(),
                    "response_time_ms": 0  # Podría medirse
                }
        except Exception as e:
            logger.error(f"❌ Error en health check NASA: {e}")
            return {
                "status": "unhealthy",
                "nasa_api_available": False,
                "api_key_configured": bool(self.api_key and self.api_key != 'DEMO_KEY'),
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def get_current_solar_activity(self) -> SolarActivitySummary:
        """
        Obtener actividad solar actual consolidada
        
        Returns:
            Resumen completo de actividad solar actual
        """
        logger.info("🔍 Obteniendo actividad solar actual...")
        
        try:
            # Obtener datos de múltiples fuentes en paralelo
            sunspot_data = await self._get_current_sunspots()
            solar_flux_data = await self._get_solar_flux()
            geomagnetic_data = await self._get_geomagnetic_indices()
            particle_data = await self._get_particle_flux()
            
            # Consolidar datos
            summary = SolarActivitySummary(
                timestamp=datetime.now(),
                sunspot_number=sunspot_data.get('sunspot_number', 0),
                solar_flux=solar_flux_data.get('flux', 0),
                kp_index=geomagnetic_data.get('kp_index', 0),
                wind_speed=geomagnetic_data.get('wind_speed', 0),
                proton_flux=particle_data.get('proton_flux', 0),
                electron_flux=particle_data.get('electron_flux', 0),
                xray_flux=await self._get_xray_flux(),
                geomagnetic_field=geomagnetic_data.get('field_status', 'quiet')
            )
            
            logger.info("✅ Actividad solar actual obtenida")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo actividad solar actual: {e}")
            # Retornar datos por defecto en caso de error
            return SolarActivitySummary(
                timestamp=datetime.now(),
                sunspot_number=0,
                solar_flux=0,
                kp_index=0,
                wind_speed=0,
                proton_flux=0,
                electron_flux=0,
                xray_flux=0,
                geomagnetic_field="unknown"
            )
    
    async def get_solar_flares(self, days: int = 7) -> List[SolarFlare]:
        """
        Obtener fulguraciones solares recientes
        
        Args:
            days: Número de días hacia atrás para buscar
            
        Returns:
            Lista de fulguraciones solares
        """
        cache_key = f"flares_{days}"
        if cache_key in self.cache and datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_duration:
            return self.cache[cache_key]['data']
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"{self.endpoints['solar_flares']}?api_key={self.api_key}&startDate={start_date}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    flares = self._parse_solar_flares(data)
                    
                    # Cachear resultados
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': flares
                    }
                    
                    logger.info(f"✅ Obtenidas {len(flares)} fulguraciones solares")
                    return flares
                else:
                    logger.warning(f"⚠️ NASA API retornó status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error obteniendo fulguraciones solares: {e}")
            return []
    
    async def get_coronal_mass_ejections(self, days: int = 7) -> List[CoronalMassEjection]:
        """
        Obtener Eyecciones de Masa Coronal recientes
        
        Args:
            days: Número de días hacia atrás para buscar
            
        Returns:
            Lista de CMEs
        """
        cache_key = f"cme_{days}"
        if cache_key in self.cache and datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_duration:
            return self.cache[cache_key]['data']
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"{self.endpoints['coronal_mass_ejections']}?api_key={self.api_key}&startDate={start_date}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    cmes = self._parse_coronal_mass_ejections(data)
                    
                    # Cachear resultados
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': cmes
                    }
                    
                    logger.info(f"✅ Obtenidas {len(cmes)} CMEs")
                    return cmes
                else:
                    logger.warning(f"⚠️ NASA API retornó status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error obteniendo CMEs: {e}")
            return []
    
    async def get_geomagnetic_storms(self, days: int = 30) -> List[GeomagneticStorm]:
        """
        Obtener tormentas geomagnéticas recientes
        
        Args:
            days: Número de días hacia atrás para buscar
            
        Returns:
            Lista de tormentas geomagnéticas
        """
        cache_key = f"storms_{days}"
        if cache_key in self.cache and datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_duration:
            return self.cache[cache_key]['data']
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"{self.endpoints['geomagnetic_storms']}?api_key={self.api_key}&startDate={start_date}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    storms = self._parse_geomagnetic_storms(data)
                    
                    # Cachear resultados
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': storms
                    }
                    
                    logger.info(f"✅ Obtenidas {len(storms)} tormentas geomagnéticas")
                    return storms
                else:
                    logger.warning(f"⚠️ NASA API retornó status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error obteniendo tormentas geomagnéticas: {e}")
            return []
    
    async def get_historical_solar_data(self, years: int = 50) -> pd.DataFrame:
        """
        Obtener datos solares históricos (manchas solares)
        
        Args:
            years: Número de años de datos históricos
            
        Returns:
            DataFrame con datos solares históricos
        """
        cache_key = f"historical_{years}"
        if cache_key in self.cache and datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(hours=1):
            return self.cache[cache_key]['data']
        
        try:
            # Usar datos del Sunspot Index and Long-term Solar Observations (SILSO)
            sunspot_data = await self._get_historical_sunspots(years)
            
            # Combinar con otros datos históricos si están disponibles
            historical_data = self._compile_historical_dataset(sunspot_data, years)
            
            # Cachear resultados
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': historical_data
            }
            
            logger.info(f"✅ Obtenidos {len(historical_data)} registros históricos solares")
            return historical_data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos históricos solares: {e}")
            # Retornar datos de ejemplo en caso de error
            return self._generate_sample_historical_data(years)
    
    async def _get_current_sunspots(self) -> Dict[str, Any]:
        """Obtener número actual de manchas solares"""
        try:
            url = f"{self.noaa_swpc_url}/sunspot_regions.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Calcular número total de manchas solares
                    sunspot_number = len(data) if isinstance(data, list) else 0
                    return {'sunspot_number': sunspot_number}
                else:
                    return {'sunspot_number': 0}
        except Exception as e:
            logger.warning(f"No se pudieron obtener manchas solares actuales: {e}")
            return {'sunspot_number': 0}
    
    async def _get_solar_flux(self) -> Dict[str, float]:
        """Obtener flujo solar actual"""
        try:
            url = f"{self.noaa_swpc_url}/f10_forecast.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Tomar el último valor de flujo pronosticado
                    flux = data[-1]['flux'] if data else 0
                    return {'flux': flux}
                else:
                    return {'flux': 0}
        except Exception as e:
            logger.warning(f"No se pudo obtener flujo solar: {e}")
            return {'flux': 0}
    
    async def _get_geomagnetic_indices(self) -> Dict[str, Any]:
        """Obtener índices geomagnéticos actuales"""
        try:
            url = f"{self.noaa_swpc_url}/geospace/geopack_1m.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Tomar los últimos valores
                    if data and len(data) > 0:
                        latest = data[-1]
                        return {
                            'kp_index': latest.get('kp', 0),
                            'wind_speed': latest.get('wind_speed', 0),
                            'field_status': self._classify_geomagnetic_field(latest.get('kp', 0))
                        }
                    else:
                        return {'kp_index': 0, 'wind_speed': 0, 'field_status': 'quiet'}
                else:
                    return {'kp_index': 0, 'wind_speed': 0, 'field_status': 'quiet'}
        except Exception as e:
            logger.warning(f"No se pudieron obtener índices geomagnéticos: {e}")
            return {'kp_index': 0, 'wind_speed': 0, 'field_status': 'quiet'}
    
    async def _get_particle_flux(self) -> Dict[str, float]:
        """Obtener flujo de partículas energéticas"""
        try:
            url = f"{self.noaa_swpc_url}/ace/swepam_1m.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        latest = data[-1]
                        return {
                            'proton_flux': latest.get('proton_density', 0),
                            'electron_flux': latest.get('electron_density', 0)
                        }
                    else:
                        return {'proton_flux': 0, 'electron_flux': 0}
                else:
                    return {'proton_flux': 0, 'electron_flux': 0}
        except Exception as e:
            logger.warning(f"No se pudo obtener flujo de partículas: {e}")
            return {'proton_flux': 0, 'electron_flux': 0}
    
    async def _get_xray_flux(self) -> float:
        """Obtener flujo de rayos X solares"""
        try:
            url = f"{self.noaa_swpc_url}/goes/primary/xrays-7-day.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        # Tomar el último valor de flujo de rayos X
                        return data[-1].get('flux', 0)
                    else:
                        return 0
                else:
                    return 0
        except Exception as e:
            logger.warning(f"No se pudo obtener flujo de rayos X: {e}")
            return 0
    
    async def _get_historical_sunspots(self, years: int) -> pd.DataFrame:
        """Obtener datos históricos de manchas solares"""
        try:
            # Descargar datos de SILSO
            async with self.session.get(self.silso_url) as response:
                if response.status == 200:
                    text_data = await response.text()
                    return self._parse_silso_data(text_data, years)
                else:
                    logger.warning("No se pudieron obtener datos históricos de SILSO")
                    return self._generate_sample_sunspot_data(years)
        except Exception as e:
            logger.warning(f"Error obteniendo datos SILSO: {e}")
            return self._generate_sample_sunspot_data(years)
    
    def _parse_silso_data(self, text_data: str, years: int) -> pd.DataFrame:
        """Parsear datos del formato SILSO"""
        lines = text_data.strip().split('\n')
        data = []
        
        for line in lines:
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        sunspots = float(parts[3])
                        
                        date = datetime(year, month, day)
                        if datetime.now().year - year <= years:
                            data.append({
                                'date': date,
                                'sunspots': sunspots,
                                'year': year,
                                'month': month
                            })
                    except (ValueError, IndexError):
                        continue
        
        return pd.DataFrame(data)
    
    def _parse_solar_flares(self, data: List[Dict]) -> List[SolarFlare]:
        """Parsear datos de fulguraciones solares de NASA DONKI"""
        flares = []
        
        for flare_data in data:
            try:
                flare = SolarFlare(
                    flare_id=flare_data.get('flrID', ''),
                    class_type=flare_data.get('classType', 'A'),
                    peak_time=datetime.fromisoformat(flare_data.get('peakTime', datetime.now().isoformat()).replace('Z', '+00:00')),
                    duration_minutes=int(flare_data.get('durationMinutes', 0)),
                    active_region=flare_data.get('activeRegionNum', ''),
                    intensity=self._calculate_flare_intensity(flare_data.get('classType', 'A')),
                    position=(0.0, 0.0)  # NASA no siempre proporciona posición
                )
                flares.append(flare)
            except Exception as e:
                logger.warning(f"Error parseando fulguración: {e}")
                continue
        
        return flares
    
    def _parse_coronal_mass_ejections(self, data: List[Dict]) -> List[CoronalMassEjection]:
        """Parsear datos de CMEs de NASA DONKI"""
        cmes = []
        
        for cme_data in data:
            try:
                # Obtener datos de actividad si están disponibles
                activity_data = cme_data.get('cmeAnalyses', [{}])[0] if cme_data.get('cmeAnalyses') else {}
                
                cme = CoronalMassEjection(
                    cme_id=cme_data.get('activityID', ''),
                    start_time=datetime.fromisoformat(cme_data.get('startTime', datetime.now().isoformat()).replace('Z', '+00:00')),
                    speed_km_s=activity_data.get('speed', 0),
                    angle_degrees=activity_data.get('latitude', 0),
                    half_angle=activity_data.get('halfAngle', 0),
                    catalog=cme_data.get('catalog', ''),
                    note=cme_data.get('note', '')
                )
                cmes.append(cme)
            except Exception as e:
                logger.warning(f"Error parseando CME: {e}")
                continue
        
        return cmes
    
    def _parse_geomagnetic_storms(self, data: List[Dict]) -> List[GeomagneticStorm]:
        """Parsear datos de tormentas geomagnéticas de NASA DONKI"""
        storms = []
        
        for storm_data in data:
            try:
                # Obtener datos del evento de tormenta
                event_data = storm_data.get('allKpIndex', [{}])[0] if storm_data.get('allKpIndex') else {}
                
                storm = GeomagneticStorm(
                    storm_id=storm_data.get('gstID', ''),
                    start_time=datetime.fromisoformat(storm_data.get('startTime', datetime.now().isoformat()).replace('Z', '+00:00')),
                    kp_index=event_data.get('kpIndex', 0),
                    gst_scale=storm_data.get('GSTMagneticCloud', 'G1'),
                    estimated_dst=storm_data.get('estimatedDST', 0),
                    cause=storm_data.get('link', '')
                )
                storms.append(storm)
            except Exception as e:
                logger.warning(f"Error parseando tormenta geomagnética: {e}")
                continue
        
        return storms
    
    def _calculate_flare_intensity(self, class_type: str) -> float:
        """Calcular intensidad numérica de fulguración solar"""
        intensity_map = {
            'A': 1.0, 'B': 2.0, 'C': 3.0, 'M': 4.0, 'X': 5.0
        }
        return intensity_map.get(class_type.upper(), 0.0)
    
    def _classify_geomagnetic_field(self, kp_index: float) -> str:
        """Clasificar estado del campo geomagnético basado en índice Kp"""
        if kp_index < 4:
            return "quiet"
        elif kp_index < 5:
            return "unsettled"
        elif kp_index < 6:
            return "active"
        elif kp_index < 7:
            return "minor storm"
        elif kp_index < 8:
            return "moderate storm"
        elif kp_index < 9:
            return "strong storm"
        else:
            return "severe storm"
    
    def _compile_historical_dataset(self, sunspot_data: pd.DataFrame, years: int) -> pd.DataFrame:
        """Compilar dataset histórico completo"""
        # Aquí se podrían agregar más fuentes de datos históricos
        # Por ahora, usar solo datos de manchas solares
        historical_data = sunspot_data.copy()
        
        # Calcular métricas derivadas
        historical_data['sunspot_ma_13'] = historical_data['sunspots'].rolling(13).mean()  # Media móvil anual aproximada
        historical_data['sunspot_ma_132'] = historical_data['sunspots'].rolling(132).mean()  # Media móvil de 11 años
        
        # Identificar ciclos solares
        historical_data['solar_cycle'] = self._identify_solar_cycles(historical_data)
        
        return historical_data
    
    def _identify_solar_cycles(self, data: pd.DataFrame) -> pd.Series:
        """Identificar ciclos solares en datos históricos"""
        # Algoritmo simple para identificar ciclos (podría mejorarse)
        cycles = []
        current_cycle = 1
        
        if len(data) == 0:
            return pd.Series([], dtype=int)
        
        for i in range(len(data)):
            if i == 0:
                cycles.append(current_cycle)
            else:
                # Cambiar ciclo cuando las manchas solares caen por debajo de un umbral y luego aumentan
                if data.iloc[i-1]['sunspots'] < 10 and data.iloc[i]['sunspots'] > 20:
                    current_cycle += 1
                cycles.append(current_cycle)
        
        return pd.Series(cycles, index=data.index)
    
    def _generate_sample_sunspot_data(self, years: int) -> pd.DataFrame:
        """Generar datos de muestra de manchas solares"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=years*365),
            end=datetime.now(),
            freq='D'
        )
        
        # Simular ciclo solar de ~11 años
        sunspots = []
        for i, date in enumerate(dates):
            # Ciclo sinusoidal de 11 años con ruido
            cycle_position = (i / (365 * 11)) * 2 * np.pi
            base_value = 50 + 40 * np.sin(cycle_position)  # 10-90 manchas
            noise = np.random.normal(0, 10)
            sunspots.append(max(0, base_value + noise))
        
        return pd.DataFrame({
            'date': dates,
            'sunspots': sunspots,
            'year': [d.year for d in dates],
            'month': [d.month for d in dates]
        })
    
    def _generate_sample_historical_data(self, years: int) -> pd.DataFrame:
        """Generar datos históricos de muestra"""
        sunspot_data = self._generate_sample_sunspot_data(years)
        return self._compile_historical_dataset(sunspot_data, years)
    
    async def get_solar_forecast(self) -> Dict[str, Any]:
        """Obtener pronóstico de actividad solar"""
        try:
            # Usar datos de NOAA para pronóstico
            url = f"{self.noaa_swpc_url}/f10_forecast.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'forecast_period': '27-day',
                        'predicted_flux': data[-1]['flux'] if data else 0,
                        'confidence': 0.7,
                        'source': 'NOAA/SWPC'
                    }
                else:
                    return {
                        'forecast_period': 'unknown',
                        'predicted_flux': 0,
                        'confidence': 0.0,
                        'source': 'fallback'
                    }
        except Exception as e:
            logger.warning(f"Error obteniendo pronóstico solar: {e}")
            return {
                'forecast_period': 'unknown',
                'predicted_flux': 0,
                'confidence': 0.0,
                'source': 'error'
            }

# Instancia global para uso en otros módulos
nasa_solar_service = NASASolarService()
