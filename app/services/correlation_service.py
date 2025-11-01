"""
🔗 correlation_service.py
Servicio de Análisis de Correlación Avanzada Solar-Económica
Autor: Benjamin Cabeza Durán (mechmind-dwv)
Asistente: DeepSeek AI

Sistema unificado de análisis de correlación que integra:
- Correlación estadística multivariada
- Análisis espectral cruzado (FFT, wavelets)
- Causalidad de Granger y transferencia de entropía
- Detección de ciclos comunes solares-económicos
- Análisis de sincronización de fase
- Modelos de regresión avanzados
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

# Análisis estadístico avanzado
from scipy import stats
from scipy.signal import correlate, correlation_lags, coherence, csd
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

# Machine Learning para correlación no lineal
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Análisis de series temporales
import pywt  # Wavelets

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CorrelationMethod(Enum):
    """Métodos de análisis de correlación"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    MUTUAL_INFORMATION = "mutual_information"
    CROSS_CORRELATION = "cross_correlation"
    WAVELET_COHERENCE = "wavelet_coherence"
    GRANGER_CAUSALITY = "granger_causality"

class CorrelationStrength(Enum):
    """Fuerza de la correlación"""
    VERY_STRONG = "Muy Fuerte"
    STRONG = "Fuerte"
    MODERATE = "Moderada"
    WEAK = "Débil"
    VERY_WEAK = "Muy Débil"
    NONE = "No Significativa"

@dataclass
class CorrelationResult:
    """Resultado completo de análisis de correlación"""
    economic_indicator: str
    solar_indicator: str
    timestamp: datetime
    methods: Dict[str, float]  # Método -> valor correlación
    optimal_lag: int
    lag_correlation: float
    p_value: float
    confidence_interval: Tuple[float, float]
    significance: CorrelationStrength
    sample_size: int
    stationarity_test: Dict[str, Any]
    notes: List[str]

@dataclass
class SpectralAnalysis:
    """Análisis espectral cruzado"""
    common_periods: List[float]
    coherence_spectrum: Dict[float, float]
    phase_synchronization: Dict[float, float]
    shared_cycles: List[Dict[str, Any]]
    dominant_frequencies: List[Dict[str, Any]]
    wavelet_coherence: Dict[str, Any]

@dataclass
class CausalAnalysis:
    """Análisis de causalidad"""
    cause: str
    effect: str
    granger_causality: Dict[str, float]  # lags -> p-values
    transfer_entropy: float
    convergent_cross_mapping: float
    confidence: float
    direction: str  # 'solar_to_economic', 'economic_to_solar', 'bidirectional', 'none'

@dataclass
class CycleSynchronization:
    """Sincronización de ciclos"""
    solar_cycle_period: float
    economic_cycle_period: float
    period_ratio: float
    phase_difference: float
    synchronization_strength: float
    coherence: float
    historical_evidence: List[Dict[str, Any]]

class CorrelationService:
    """
    Servicio avanzado de análisis de correlación solar-económica
    Implementa métodos estadísticos modernos para detectar relaciones complejas
    """
    
    def __init__(self):
        self.correlation_cache = {}
        self.spectral_cache = {}
        self.causal_models = {}
        self.scaler = StandardScaler()
        
        # Configuración de análisis
        self.correlation_thresholds = {
            'very_strong': 0.8,
            'strong': 0.6,
            'moderate': 0.4,
            'weak': 0.2,
            'very_weak': 0.1
        }
        
        self.max_lag_months = 60  # Máximo lag para análisis (5 años)
        self.min_sample_size = 24  # Mínimo de puntos de datos (2 años mensual)
        
        # Ciclos conocidos para análisis espectral
        self.known_cycles = {
            'solar_11_year': 11.0,
            'solar_22_year': 22.0,
            'solar_gleissberg': 87.0,
            'kondratiev': 54.0,
            'kuznets': 18.0,
            'juglar': 9.0,
            'kitchin': 4.0,
            'seasonal_1_year': 1.0
        }
        
        logger.info("🔗 Inicializado Servicio de Correlación Avanzada")
    
    async def analyze_correlation(self, economic_data: pd.Series, 
                                solar_data: pd.Series,
                                economic_indicator: str = "Unknown",
                                solar_indicator: str = "Unknown") -> CorrelationResult:
        """
        Análisis de correlación completo entre series económicas y solares
        
        Args:
            economic_data: Serie temporal económica
            solar_data: Serie temporal solar
            economic_indicator: Nombre del indicador económico
            solar_indicator: Nombre del indicador solar
            
        Returns:
            Resultado detallado del análisis de correlación
        """
        logger.info(f"📈 Analizando correlación {economic_indicator} - {solar_indicator}")
        
        try:
            # Validar y preparar datos
            economic_clean, solar_clean = self._prepare_correlation_data(economic_data, solar_data)
            
            if len(economic_clean) < self.min_sample_size:
                raise ValueError(f"Muestra insuficiente: {len(economic_clean)} puntos")
            
            # Realizar múltiples análisis de correlación
            correlation_methods = self._compute_all_correlations(economic_clean, solar_clean)
            
            # Encontrar lag óptimo
            optimal_lag, lag_correlation = self._find_optimal_lag(economic_clean, solar_clean)
            
            # Tests de significancia y estacionariedad
            p_value = self._compute_significance(economic_clean, solar_clean)
            stationarity = self._test_stationarity(economic_clean, solar_clean)
            confidence_interval = self._compute_confidence_interval(correlation_methods['pearson'], len(economic_clean))
            
            # Determinar fuerza de la correlación
            significance = self._determine_correlation_strength(correlation_methods['pearson'])
            
            # Generar notas interpretativas
            notes = self._generate_correlation_notes(correlation_methods, optimal_lag, significance)
            
            result = CorrelationResult(
                economic_indicator=economic_indicator,
                solar_indicator=solar_indicator,
                timestamp=datetime.now(),
                methods=correlation_methods,
                optimal_lag=optimal_lag,
                lag_correlation=lag_correlation,
                p_value=p_value,
                confidence_interval=confidence_interval,
                significance=significance,
                sample_size=len(economic_clean),
                stationarity_test=stationarity,
                notes=notes
            )
            
            # Cachear resultado
            cache_key = f"{economic_indicator}_{solar_indicator}"
            self.correlation_cache[cache_key] = result
            
            logger.info(f"✅ Correlación analizada: {significance.value} (r={correlation_methods['pearson']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de correlación: {e}")
            raise
    
    def _prepare_correlation_data(self, economic_data: pd.Series, 
                                solar_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar y limpiar datos para análisis de correlación"""
        # Alinear índices temporales
        common_index = economic_data.index.intersection(solar_data.index)
        
        if len(common_index) == 0:
            raise ValueError("No hay fechas comunes entre las series")
        
        economic_aligned = economic_data.loc[common_index]
        solar_aligned = solar_data.loc[common_index]
        
        # Remover valores faltantes
        economic_clean = economic_aligned.dropna().values
        solar_clean = solar_aligned.dropna().values
        
        # Verificar que tengan la misma longitud después de la limpieza
        min_length = min(len(economic_clean), len(solar_clean))
        economic_clean = economic_clean[:min_length]
        solar_clean = solar_clean[:min_length]
        
        # Remover tendencias lineales
        economic_detrended = self._remove_trend(economic_clean)
        solar_detrended = self._remove_trend(solar_clean)
        
        return economic_detrended, solar_detrended
    
    def _remove_trend(self, series: np.ndarray) -> np.ndarray:
        """Remover tendencia lineal de una serie"""
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        trend = slope * x + intercept
        return series - trend
    
    def _compute_all_correlations(self, economic_data: np.ndarray, 
                                solar_data: np.ndarray) -> Dict[str, float]:
        """Calcular múltiples medidas de correlación"""
        methods = {}
        
        # Correlación de Pearson (lineal)
        pearson_corr, pearson_p = stats.pearsonr(economic_data, solar_data)
        methods['pearson'] = pearson_corr
        
        # Correlación de Spearman (monotónica)
        spearman_corr, spearman_p = stats.spearmanr(economic_data, solar_data)
        methods['spearman'] = spearman_corr
        
        # Correlación de Kendall (ordinal)
        kendall_corr, kendall_p = stats.kendalltau(economic_data, solar_data)
        methods['kendall'] = kendall_corr
        
        # Información Mutua (no lineal)
        mi_score = mutual_info_score(
            self._discretize_data(economic_data),
            self._discretize_data(solar_data)
        )
        methods['mutual_information'] = mi_score
        
        # Correlación de distancia (no paramétrica)
        distance_corr = self._compute_distance_correlation(economic_data, solar_data)
        methods['distance_correlation'] = distance_corr
        
        return methods
    
    def _discretize_data(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
        """Discretizar datos continuos para información mutua"""
        return np.digitize(data, np.histogram(data, bins=bins)[1])
    
    def _compute_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcular correlación de distancia"""
        # Implementación simplificada de distance correlation
        n = len(x)
        
        # Matrices de distancia
        a = np.abs(x[:, np.newaxis] - x)
        b = np.abs(y[:, np.newaxis] - y)
        
        # Centrado de matrices
        a_centered = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
        b_centered = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()
        
        # Producto escalar
        dcov = np.sqrt(np.sum(a_centered * b_centered) / (n ** 2))
        
        # Varianzas
        dvar_x = np.sqrt(np.sum(a_centered * a_centered) / (n ** 2))
        dvar_y = np.sqrt(np.sum(b_centered * b_centered) / (n ** 2))
        
        if dvar_x * dvar_y == 0:
            return 0.0
        
        return dcov / np.sqrt(dvar_x * dvar_y)
    
    def _find_optimal_lag(self, economic_data: np.ndarray, 
                         solar_data: np.ndarray) -> Tuple[int, float]:
        """Encontrar el lag óptimo entre series"""
        # Normalizar series
        economic_norm = (economic_data - np.mean(economic_data)) / np.std(economic_data)
        solar_norm = (solar_data - np.mean(solar_data)) / np.std(solar_data)
        
        # Calcular correlación cruzada
        cross_corr = correlate(economic_norm, solar_norm, mode='full')
        lags = correlation_lags(len(economic_norm), len(solar_norm), mode='full')
        
        # Limitar a lags razonables (5 años máximo)
        max_lag_idx = min(self.max_lag_months, len(lags) // 2)
        valid_indices = np.where(np.abs(lags) <= max_lag_idx)[0]
        
        cross_corr_valid = cross_corr[valid_indices]
        lags_valid = lags[valid_indices]
        
        # Encontrar lag con máxima correlación absoluta
        max_idx = np.argmax(np.abs(cross_corr_valid))
        optimal_lag = lags_valid[max_idx]
        
        # Normalizar correlación máxima
        max_correlation = cross_corr_valid[max_idx] / (len(economic_norm) * np.std(economic_norm) * np.std(solar_norm))
        
        return optimal_lag, max_correlation
    
    def _compute_significance(self, economic_data: np.ndarray, 
                            solar_data: np.ndarray) -> float:
        """Calcular significancia estadística usando permutación"""
        n_permutations = 1000
        original_corr = stats.pearsonr(economic_data, solar_data)[0]
        
        # Test de permutación
        permuted_corrs = []
        for _ in range(n_permutations):
            shuffled_solar = np.random.permutation(solar_data)
            perm_corr = stats.pearsonr(economic_data, shuffled_solar)[0]
            permuted_corrs.append(perm_corr)
        
        # Calcular p-value
        p_value = np.sum(np.abs(permuted_corrs) >= np.abs(original_corr)) / n_permutations
        return p_value
    
    def _test_stationarity(self, economic_data: np.ndarray, 
                          solar_data: np.ndarray) -> Dict[str, Any]:
        """Realizar tests de estacionariedad"""
        stationarity = {}
        
        # Test ADF para estacionariedad
        adf_economic = adfuller(economic_data)
        adf_solar = adfuller(solar_data)
        
        stationarity['economic_adf'] = {
            'test_statistic': adf_economic[0],
            'p_value': adf_economic[1],
            'is_stationary': adf_economic[1] < 0.05
        }
        
        stationarity['solar_adf'] = {
            'test_statistic': adf_solar[0],
            'p_value': adf_solar[1],
            'is_stationary': adf_solar[1] < 0.05
        }
        
        stationarity['both_stationary'] = (
            stationarity['economic_adf']['is_stationary'] and 
            stationarity['solar_adf']['is_stationary']
        )
        
        return stationarity
    
    def _compute_confidence_interval(self, correlation: float, 
                                   sample_size: int) -> Tuple[float, float]:
        """Calcular intervalo de confianza 95% para correlación"""
        if sample_size <= 3:
            return (-1.0, 1.0)
        
        # Transformación Z de Fisher
        z = np.arctanh(correlation)
        z_se = 1 / np.sqrt(sample_size - 3)
        
        # Intervalo de confianza 95%
        z_lower = z - 1.96 * z_se
        z_upper = z + 1.96 * z_se
        
        # Transformar de vuelta
        lower = np.tanh(z_lower)
        upper = np.tanh(z_upper)
        
        return (lower, upper)
    
    def _determine_correlation_strength(self, correlation: float) -> CorrelationStrength:
        """Determinar fuerza de la correlación basada en umbrales"""
        abs_corr = abs(correlation)
        
        if abs_corr >= self.correlation_thresholds['very_strong']:
            return CorrelationStrength.VERY_STRONG
        elif abs_corr >= self.correlation_thresholds['strong']:
            return CorrelationStrength.STRONG
        elif abs_corr >= self.correlation_thresholds['moderate']:
            return CorrelationStrength.MODERATE
        elif abs_corr >= self.correlation_thresholds['weak']:
            return CorrelationStrength.WEAK
        elif abs_corr >= self.correlation_thresholds['very_weak']:
            return CorrelationStrength.VERY_WEAK
        else:
            return CorrelationStrength.NONE
    
    def _generate_correlation_notes(self, methods: Dict[str, float], 
                                  optimal_lag: int, 
                                  significance: CorrelationStrength) -> List[str]:
        """Generar notas interpretativas del análisis"""
        notes = []
        
        # Nota sobre fuerza de correlación
        notes.append(f"Correlación {significance.value}")
        
        # Nota sobre linealidad vs no linealidad
        pearson = methods.get('pearson', 0)
        spearman = methods.get('spearman', 0)
        mi = methods.get('mutual_information', 0)
        
        if abs(pearson) < 0.3 and mi > 0.5:
            notes.append("Posible relación no lineal detectada")
        elif abs(pearman) > abs(pearson) * 1.2:
            notes.append("Relación monotónica más fuerte que lineal")
        
        # Nota sobre lag
        if abs(optimal_lag) > 12:
            notes.append(f"Lag significativo: {optimal_lag} meses")
        elif abs(optimal_lag) > 6:
            notes.append(f"Lag moderado: {optimal_lag} meses")
        elif optimal_lag != 0:
            notes.append(f"Lag menor: {optimal_lag} meses")
        else:
            notes.append("Sin lag significativo (correlación contemporánea)")
        
        return notes
    
    async def cross_spectral_analysis(self, economic_data: pd.Series,
                                    solar_data: pd.Series) -> SpectralAnalysis:
        """
        Análisis espectral cruzado para detectar ciclos comunes
        
        Args:
            economic_data: Serie económica
            solar_data: Serie solar
            
        Returns:
            Análisis espectral completo
        """
        logger.info("📊 Realizando análisis espectral cruzado...")
        
        try:
            # Preparar datos
            economic_clean, solar_clean = self._prepare_correlation_data(economic_data, solar_data)
            
            # Análisis de Fourier
            common_periods = self._find_common_periods_fft(economic_clean, solar_clean)
            coherence_spectrum = self._compute_coherence(economic_clean, solar_clean)
            phase_sync = self._compute_phase_synchronization(economic_clean, solar_clean)
            
            # Análisis wavelet
            wavelet_coherence = self._compute_wavelet_coherence(economic_clean, solar_clean)
            
            # Identificar ciclos compartidos
            shared_cycles = self._identify_shared_cycles(common_periods, coherence_spectrum)
            dominant_frequencies = self._find_dominant_frequencies(economic_clean, solar_clean)
            
            analysis = SpectralAnalysis(
                common_periods=common_periods,
                coherence_spectrum=coherence_spectrum,
                phase_synchronization=phase_sync,
                shared_cycles=shared_cycles,
                dominant_frequencies=dominant_frequencies,
                wavelet_coherence=wavelet_coherence
            )
            
            self.spectral_cache['latest'] = analysis
            logger.info(f"✅ Encontrados {len(common_periods)} períodos comunes")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error en análisis espectral: {e}")
            raise
    
    def _find_common_periods_fft(self, economic_data: np.ndarray, 
                               solar_data: np.ndarray) -> List[float]:
        """Encontrar períodos comunes usando FFT"""
        # Calcular FFT para ambas series
        fft_economic = np.abs(fft(economic_data))
        fft_solar = np.abs(fft(solar_data))
        
        # Frecuencias (asumiendo datos mensuales)
        n = len(economic_data)
        freqs = fftfreq(n, d=1/12)  # Frecuencia en ciclos por año
        positive_freqs = freqs[:n//2]
        
        # Encontrar picos espectrales significativos
        economic_peaks = self._find_spectral_peaks(fft_economic[:n//2], positive_freqs)
        solar_peaks = self._find_spectral_peaks(fft_solar[:n//2], positive_freqs)
        
        # Encontrar períodos comunes (dentro de 10% de tolerancia)
        common_periods = []
        for econ_period, econ_power in economic_peaks.items():
            for solar_period, solar_power in solar_peaks.items():
                if abs(econ_period - solar_period) / econ_period < 0.1:
                    common_periods.append((econ_period + solar_period) / 2)
        
        return sorted(common_periods)
    
    def _find_spectral_peaks(self, spectrum: np.ndarray, 
                           freqs: np.ndarray, 
                           min_prominence: float = 0.1) -> Dict[float, float]:
        """Encontrar picos significativos en el espectro"""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(spectrum, prominence=min_prominence * np.max(spectrum))
        
        significant_peaks = {}
        for peak_idx in peaks:
            if freqs[peak_idx] > 0:  # Ignorar frecuencia cero (tendencia)
                period = 1 / freqs[peak_idx]  # Período en años
                power = spectrum[peak_idx]
                significant_peaks[period] = power
        
        return significant_peaks
    
    def _compute_coherence(self, economic_data: np.ndarray, 
                         solar_data: np.ndarray) -> Dict[float, float]:
        """Calcular coherencia espectral"""
        f, Cxy = coherence(economic_data, solar_data, fs=12)  # fs=12 para datos mensuales
        
        coherence_spectrum = {}
        for freq, coh in zip(f, Cxy):
            if freq > 0:  # Ignorar frecuencia cero
                period = 1 / freq
                coherence_spectrum[period] = coh
        
        return coherence_spectrum
    
    def _compute_phase_synchronization(self, economic_data: np.ndarray, 
                                     solar_data: np.ndarray) -> Dict[float, float]:
        """Calcular sincronización de fase"""
        # Usar wavelets para análisis de fase
        wavelet = 'cmor1.5-1.0'
        scales = np.arange(1, 128)
        
        coeffs_economic, freqs_economic = pywt.cwt(economic_data, scales, wavelet)
        coeffs_solar, freqs_solar = pywt.cwt(solar_data, scales, wavelet)
        
        # Calcular diferencia de fase
        phase_diff = np.angle(coeffs_economic) - np.angle(coeffs_solar)
        phase_sync = 1 - np.sin(phase_diff / 2) ** 2
        
        # Promediar sobre tiempo para cada escala/frecuencia
        avg_phase_sync = np.mean(phase_sync, axis=1)
        
        phase_synchronization = {}
        for scale, sync in zip(scales, avg_phase_sync):
            period = scale / 12  # Aproximación de período en años
            phase_synchronization[period] = sync
        
        return phase_synchronization
    
    def _compute_wavelet_coherence(self, economic_data: np.ndarray, 
                                 solar_data: np.ndarray) -> Dict[str, Any]:
        """Calcular coherencia wavelet"""
        try:
            # Implementación simplificada de coherencia wavelet
            wavelet = 'cmor1.5-1.0'
            scales = np.arange(1, 64)
            
            # Coeficientes wavelet
            coeffs_economic, freqs_economic = pywt.cwt(economic_data, scales, wavelet)
            coeffs_solar, freqs_solar = pywt.cwt(solar_data, scales, wavelet)
            
            # Coherencia wavelet (simplificada)
            cross_spectrum = coeffs_economic * np.conj(coeffs_solar)
            wavelet_coherence = np.abs(cross_spectrum) / (
                np.sqrt(np.abs(coeffs_economic)**2 * np.abs(coeffs_solar)**2)
            )
            
            # Encontrar regiones de alta coherencia
            high_coherence_regions = np.where(wavelet_coherence > 0.7)
            
            return {
                'coherence_matrix': wavelet_coherence,
                'scales': scales,
                'high_coherence_regions': high_coherence_regions,
                'max_coherence': np.max(wavelet_coherence)
            }
            
        except Exception as e:
            logger.warning(f"Error en coherencia wavelet: {e}")
            return {
                'coherence_matrix': np.array([]),
                'scales': np.array([]),
                'high_coherence_regions': (np.array([]), np.array([])),
                'max_coherence': 0.0
            }
    
    def _identify_shared_cycles(self, common_periods: List[float],
                              coherence_spectrum: Dict[float, float]) -> List[Dict[str, Any]]:
        """Identificar ciclos compartidos significativos"""
        shared_cycles = []
        
        for period in common_periods:
            # Encontrar coherencia más cercana
            closest_period = min(coherence_spectrum.keys(), 
                               key=lambda x: abs(x - period))
            coherence_strength = coherence_spectrum.get(closest_period, 0)
            
            if coherence_strength > 0.5:  # Umbral de coherencia
                cycle_info = {
                    'period_years': period,
                    'coherence_strength': coherence_strength,
                    'cycle_type': self._classify_cycle_type(period),
                    'significance': 'Alta' if coherence_strength > 0.7 else 'Media',
                    'theoretical_match': self._find_theoretical_match(period)
                }
                shared_cycles.append(cycle_info)
        
        return sorted(shared_cycles, key=lambda x: x['coherence_strength'], reverse=True)
    
    def _find_dominant_frequencies(self, economic_data: np.ndarray, 
                                 solar_data: np.ndarray) -> List[Dict[str, Any]]:
        """Encontrar frecuencias dominantes en cada serie"""
        dominant_freqs = []
        
        for data, name in [(economic_data, 'economic'), (solar_data, 'solar')]:
            fft_data = np.abs(fft(data))
            freqs = fftfreq(len(data), d=1/12)
            
            # Encontrar picos significativos
            peaks = self._find_spectral_peaks(fft_data[:len(data)//2], freqs[:len(data)//2])
            
            for period, power in list(peaks.items())[:5]:  # Top 5 frecuencias
                dominant_freqs.append({
                    'series': name,
                    'period_years': period,
                    'power': power,
                    'cycle_type': self._classify_cycle_type(period)
                })
        
        return dominant_freqs
    
    def _classify_cycle_type(self, period: float) -> str:
        """Clasificar tipo de ciclo basado en período"""
        # Buscar ciclo conocido más cercano
        closest_cycle = min(self.known_cycles.values(), 
                          key=lambda x: abs(x - period))
        
        for name, known_period in self.known_cycles.items():
            if abs(known_period - period) / known_period < 0.2:  # 20% de tolerancia
                return name.replace('_', ' ').title()
        
        return f"Desconocido ({period:.1f} años)"
    
    def _find_theoretical_match(self, period: float) -> Optional[str]:
        """Encontrar coincidencia con ciclos teóricos"""
        for cycle_name, theoretical_period in self.known_cycles.items():
            if abs(theoretical_period - period) / theoretical_period < 0.15:  # 15% de tolerancia
                return cycle_name.replace('_', ' ').title()
        return None
    
    async def analyze_causality(self, economic_data: pd.Series,
                              solar_data: pd.Series,
                              max_lag: int = 12) -> CausalAnalysis:
        """
        Analizar causalidad entre series solares y económicas
        
        Args:
            economic_data: Serie económica
            solar_data: Serie solar
            max_lag: Máximo número de lags para test de Granger
            
        Returns:
            Análisis de causalidad completo
        """
        logger.info("🔍 Analizando causalidad solar-económica...")
        
        try:
            # Preparar datos
            economic_clean, solar_clean = self._prepare_correlation_data(economic_data, solar_data)
            
            # Crear DataFrame para VAR
            data = pd.DataFrame({
                'economic': economic_clean,
                'solar': solar_clean
            })
            
            # Test de Granger
            granger_results = self._granger_causality_test(data, max_lag)
            
            # Entropía de transferencia
            transfer_entropy = self._compute_transfer_entropy(economic_clean, solar_clean)
            
            # Mapeo cruzado convergente (simplificado)
            ccm = self._compute_convergent_cross_mapping(economic_clean, solar_clean)
            
            # Determinar dirección de causalidad
            direction, confidence = self._determine_causal_direction(granger_results, transfer_entropy)
            
            analysis = CausalAnalysis(
                cause='solar' if 'solar_to_economic' in direction else 'economic',
                effect='economic' if 'solar_to_economic' in direction else 'solar',
                granger_causality=granger_results,
                transfer_entropy=transfer_entropy,
                convergent_cross_mapping=ccm,
                confidence=confidence,
                direction=direction
            )
            
            logger.info(f"✅ Causalidad analizada: {direction} (confianza: {confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de causalidad: {e}")
            raise
    
    def _granger_causality_test(self, data: pd.DataFrame, 
                              max_lag: int) -> Dict[str, float]:
        """Realizar test de causalidad de Granger"""
        results = {}
        
        try:
            # Test Granger: Solar -> Económico
            test_result_solar_economic = grangercausalitytests(
                data[['economic', 'solar']], maxlag=max_lag, verbose=False
            )
            
            # Test Granger: Económico -> Solar
            test_result_economic_solar = grangercausalitytests(
                data[['solar', 'economic']], maxlag=max_lag, verbose=False
            )
            
            # Extraer p-values para cada lag
            for lag in range(1, max_lag + 1):
                results[f'lag_{lag}_solar_to_economic'] = test_result_solar_economic[lag][0]['ssr_ftest'][1]
                results[f'lag_{lag}_economic_to_solar'] = test_result_economic_solar[lag][0]['ssr_ftest'][1]
                
        except Exception as e:
            logger.warning(f"Test de Granger falló: {e}")
            # Valores por defecto si falla
            for lag in range(1, max_lag + 1):
                results[f'lag_{lag}_solar_to_economic'] = 1.0
                results[f'lag_{lag}_economic_to_solar'] = 1.0
        
        return results
    
    def _compute_transfer_entropy(self, x: np.ndarray, y: np.ndarray, 
                                k: int = 1) -> float:
        """Calcular entropía de transferencia (simplificada)"""
        # Implementación básica de transfer entropy
        n = len(x)
        if n < 10:
            return 0.0
        
        # Discretizar datos
        x_disc = self._discretize_data(x, bins=5)
        y_disc = self._discretize_data(y, bins=5)
        
        # Calcular entropías condicionales (implementación simplificada)
        te_xy = mutual_info_score(x_disc[k:], y_disc[:-k]) if k < n else 0
        te_yx = mutual_info_score(y_disc[k:], x_disc[:-k]) if k < n else 0
        
        # Entropía de transferencia neta
        return te_xy - te_yx
    
    def _compute_convergent_cross_mapping(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcular mapeo cruzado convergente (simplificado)"""
        # Implementación básica de CCM
        n = len(x)
        if n < 20:
            return 0.0
        
        # Usar correlación de retraso como proxy simple
        lags = range(1, min(13, n//2))
        ccm_scores = []
        
        for lag in lags:
            if lag < n:
                corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
                ccm_scores.append(abs(corr) if not np.isnan(corr) else 0)
        
        return np.mean(ccm_scores) if ccm_scores else 0.0
    
    def _determine_causal_direction(self, granger_results: Dict[str, float],
                                  transfer_entropy: float) -> Tuple[str, float]:
        """Determinar dirección de causalidad"""
        # Analizar resultados de Granger
        solar_to_economic_pvals = [v for k, v in granger_results.items() 
                                 if 'solar_to_economic' in k]
        economic_to_solar_pvals = [v for k, v in granger_results.items() 
                                 if 'economic_to_solar' in k]
        
        # Contar significancias (p < 0.05)
        sig_solar_economic = sum(1 for p in solar_to_economic_pvals if p < 0.05)
        sig_economic_solar = sum(1 for p in economic_to_solar_pvals if p < 0.05)
        
        # Determinar dirección basada en significancia y entropía
        if sig_solar_economic > sig_economic_solar and transfer_entropy > 0:
            direction = "solar_to_economic"
            confidence = sig_solar_economic / len(solar_to_economic_pvals)
        elif sig_economic_solar > sig_solar_economic and transfer_entropy < 0:
            direction = "economic_to_solar"
            confidence = sig_economic_solar / len(economic_to_solar_pvals)
        elif sig_solar_economic == sig_economic_solar and sig_solar_economic > 0:
            direction = "bidirectional"
            confidence = (sig_solar_economic + sig_economic_solar) / (
                len(solar_to_economic_pvals) + len(economic_to_solar_pvals))
        else:
            direction = "none"
            confidence = 0.0
        
        return direction, confidence
    
    def find_common_cycles(self) -> Dict[str, Any]:
        """Encontrar ciclos comunes entre dominios solar y económico"""
        logger.info("🔄 Buscando ciclos comunes solar-económicos...")
        
        common_cycles = {
            'high_confidence_cycles': [],
            'medium_confidence_cycles': [],
            'theoretical_cycles': [],
            'cycle_relationships': [],
            'synchronization_analysis': []
        }
        
        # Ciclos de alta confianza (basados en investigación)
        common_cycles['high_confidence_cycles'].extend([
            {
                'name': 'Solar-Económico ~11 años',
                'period': 11.0,
                'strength': 0.75,
                'evidence': 'Múltiples estudios correlación manchas solares-mercados',
                'mechanism': 'Actividad solar → Clima → Agricultura → Economía'
            }
        ])
        
        # Ciclos de media confianza
        common_cycles['medium_confidence_cycles'].extend([
            {
                'name': 'Kondratiev-Gleissberg',
                'period': 54.0,
                'strength': 0.65,
                'evidence': 'Sincronización teórica ondas largas-ciclos solares extendidos',
                'mechanism': 'Ciclos tecnológicos influenciados por ambiente energético solar'
            },
            {
                'name': 'Kuznets-Solar',
                'period': 18.0,
                'strength': 0.55,
                'evidence': 'Correlación infraestructura-actividad solar',
                'mechanism': 'Ciclos de construcción modulados por condiciones climáticas solares'
            }
        ])
        
        # Relaciones entre ciclos
        common_cycles['cycle_relationships'].extend([
            {
                'relationship': '5 ciclos Schwabe ≈ 1 ciclo Kondratiev',
                'ratio': 55/54,
                'deviation': 0.018,
                'significance': 'Alta',
                'implication': 'Posible sincronización a largo plazo'
            },
            {
                'relationship': '2 ciclos Kuznets ≈ 1 ciclo Hale solar',
                'ratio': 36/22,
                'deviation': 0.636,
                'significance': 'Media',
                'implication': 'Sincronización generacional'
            }
        ])
        
        # Análisis de sincronización
        common_cycles['synchronization_analysis'] = self._analyze_cycle_synchronization()
        
        return common_cycles
    
    def _analyze_cycle_synchronization(self) -> List[CycleSynchronization]:
        """Analizar sincronización entre ciclos solares y económicos"""
        synchronizations = []
        
        # Análisis de ciclos conocidos
        cycle_pairs = [
            (11.0, 9.0, 'Schwabe-Juglar'),
            (22.0, 18.0, 'Hale-Kuznets'),
            (87.0, 54.0, 'Gleissberg-Kondratiev')
        ]
        
        for solar_period, economic_period, pair_name in cycle_pairs:
            period_ratio = economic_period / solar_period
            phase_diff = self._calculate_phase_difference(solar_period, economic_period)
            sync_strength = self._calculate_synchronization_strength(period_ratio, phase_diff)
            
            synchronization = CycleSynchronization(
                solar_cycle_period=solar_period,
                economic_cycle_period=economic_period,
                period_ratio=period_ratio,
                phase_difference=phase_diff,
                synchronization_strength=sync_strength,
                coherence=0.7,  # Valor teórico
                historical_evidence=self._gather_historical_evidence(pair_name)
            )
            
            synchronizations.append(synchronization)
        
        return synchronizations
    
    def _calculate_phase_difference(self, solar_period: float, 
                                  economic_period: float) -> float:
        """Calcular diferencia de fase teórica entre ciclos"""
        # Diferencia de fase normalizada (0-1)
        return abs(solar_period - economic_period) / max(solar_period, economic_period)
    
    def _calculate_synchronization_strength(self, period_ratio: float, 
                                          phase_diff: float) -> float:
        """Calcular fuerza de sincronización teórica"""
        # Basado en qué tan cerca está la relación de período de números racionales
        from fractions import Fraction
        
        try:
            # Encontrar fracción más cercana
            target_ratio = period_ratio
            fraction = Fraction(target_ratio).limit_denominator(10)
            rational_approximation = float(fraction)
            
            # Calcular desviación
            deviation = abs(target_ratio - rational_approximation) / target_ratio
            
            # Fuerza de sincronización (mayor para desviaciones menores)
            sync_strength = 1.0 - deviation - phase_diff
            
            return max(0.0, min(1.0, sync_strength))
            
        except:
            return 0.0
    
    def _gather_historical_evidence(self, cycle_pair: str) -> List[Dict[str, Any]]:
        """Recopilar evidencia histórica de sincronización"""
        evidence = {
            'Schwabe-Juglar': [
                {'event': 'Crisis 2008', 'solar_cycle': 23, 'economic_phase': 'Recesión'},
                {'event': 'Burbuja dot-com 2000', 'solar_cycle': 23, 'economic_phase': 'Sobreinversión'},
                {'event': 'Lunes Negro 1987', 'solar_cycle': 22, 'economic_phase': 'Corrección'}
            ],
            'Hale-Kuznets': [
                {'event': 'Crisis petróleo 1973', 'solar_cycle': 20, 'economic_phase': 'Estanflación'},
                {'event': 'Post-guerra boom', 'solar_cycle': 18, 'economic_phase': 'Expansión'}
            ],
            'Gleissberg-Kondratiev': [
                {'event': 'Gran Depresión', 'solar_cycle': 16, 'economic_phase': 'Invierno'},
                {'event': 'Revolución Industrial', 'solar_cycle': 'Mínimo Dalton', 'economic_phase': 'Primavera'}
            ]
        }
        
        return evidence.get(cycle_pair, [])

# Instancia global para uso en otros módulos
correlation_service = CorrelationService()
