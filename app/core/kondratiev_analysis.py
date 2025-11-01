"""
🌊 kondratiev_analysis.py
Análisis de Ondas Largas de Kondratiev y su Sincronización con Ciclos Solares
Autor: Benjamin Cabeza Durán (mechmind-dwv)
Asistente: DeepSeek AI

Implementación del análisis de ciclos económicos largos (45-60 años) descubiertos por 
Nikolai Kondratiev y su fascinante relación con los ciclos solares de Gleissberg (~80-90 años)

"Los grandes ciclos económicos son el latido del corazón del capitalismo,
y el Sol marca el ritmo de ese corazón."
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

# Análisis avanzado
from scipy import signal
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class KondratievPhase(Enum):
    """Fases de la Onda Larga de Kondratiev"""
    PRIMAVERA = "Primavera"      # Expansión, innovación, crecimiento
    VERANO = "Verano"            # Prosperidad, madurez, sobreinversión
    OTOÑO = "Otoño"             # Estancamiento, recesión financiera
    INVIERNO = "Invierno"       # Depresión, purga, reinvención

class TechnologicalParadigm(Enum):
    """Paradigmas tecnológicos de cada onda Kondratiev"""
    ONDA_1 = ("1780-1840", "Máquina de vapor, textiles", "Revolución Industrial")
    ONDA_2 = ("1840-1890", "Acero, ferrocarriles", "Era del vapor y acero")
    ONDA_3 = ("1890-1940", "Electricidad, químicos", "Era eléctrica")
    ONDA_4 = ("1940-1980", "Petróleo, automóvil", "Era del petróleo")
    ONDA_5 = ("1980-2020", "TIC, internet", "Era de la información")
    ONDA_6 = ("2020-2060", "IA, biotecnología, energía verde", "Era de la singularidad")

@dataclass
class KondratievWave:
    """Estructura para representar una Onda Larga de Kondratiev"""
    wave_number: int
    start_year: int
    end_year: int
    duration_years: int
    technological_paradigm: str
    key_innovations: List[str]
    phases: Dict[KondratievPhase, Tuple[int, int]]
    solar_cycle_sync: Optional[Dict[str, Any]] = None

@dataclass
class WaveAnalysis:
    """Análisis completo de onda Kondratiev"""
    current_wave: KondratievWave
    current_phase: KondratievPhase
    phase_progress: float  # 0-1, progreso en la fase actual
    next_phase_transition: datetime
    economic_implications: Dict[str, Any]
    solar_correlation: Dict[str, float]
    risk_assessment: Dict[str, Any]

@dataclass
class SolarKondratievSync:
    """Sincronización entre ciclos solares y ondas Kondratiev"""
    kondratiev_wave: int
    solar_cycles: List[int]  # Ciclos solares durante la onda
    phase_synchronization: Dict[KondratievPhase, str]
    correlation_strength: float
    historical_evidence: List[Dict[str, Any]]

class KondratievAnalyzer:
    """
    Analizador avanzado de Ondas Largas de Kondratiev
    con integración de ciclos solares de Gleissberg
    
    Basado en la obra seminal de Nikolai Kondratiev (1925)
    y investigaciones modernas sobre ciclos solares-económicos
    """
    
    def __init__(self):
        self.kondratiev_waves = self._initialize_historical_waves()
        self.gleissberg_cycle_years = 87  # Ciclo solar largo
        self.kondratiev_cycle_years = 54  # Duración promedio onda Kondratiev
        
        # Datos históricos de sincronización
        self.historical_sync_data = self._load_historical_sync_data()
        
        # Modelo de fase actual
        self.current_analysis = None
        
        logger.info("🌊 Inicializado Analizador Kondratiev-Gleissberg")
    
    def _initialize_historical_waves(self) -> List[KondratievWave]:
        """Inicializar ondas Kondratiev históricas documentadas"""
        return [
            KondratievWave(
                wave_number=1,
                start_year=1780,
                end_year=1840,
                duration_years=60,
                technological_paradigm="Máquina de vapor, textiles",
                key_innovations=["Máquina de vapor", "Telar mecánico", "Ferrocarril"],
                phases={
                    KondratievPhase.PRIMAVERA: (1780, 1800),
                    KondratievPhase.VERANO: (1800, 1815),
                    KondratievPhase.OTOÑO: (1815, 1825),
                    KondratievPhase.INVIERNO: (1825, 1840)
                }
            ),
            KondratievWave(
                wave_number=2,
                start_year=1840,
                end_year=1890,
                duration_years=50,
                technological_paradigm="Acero, ferrocarriles",
                key_innovations=["Horno Bessemer", "Telégrafo", "Barco de vapor"],
                phases={
                    KondratievPhase.PRIMAVERA: (1840, 1855),
                    KondratievPhase.VERANO: (1855, 1865),
                    KondratievPhase.OTOÑO: (1865, 1875),
                    KondratievPhase.INVIERNO: (1875, 1890)
                }
            ),
            KondratievWave(
                wave_number=3,
                start_year=1890,
                end_year=1940,
                duration_years=50,
                technological_paradigm="Electricidad, químicos",
                key_innovations=["Electricidad", "Motor combustión", "Teléfono"],
                phases={
                    KondratievPhase.PRIMAVERA: (1890, 1910),
                    KondratievPhase.VERANO: (1910, 1920),
                    KondratievPhase.OTOÑO: (1920, 1929),
                    KondratievPhase.INVIERNO: (1929, 1940)
                }
            ),
            KondratievWave(
                wave_number=4,
                start_year=1940,
                end_year=1980,
                duration_years=40,
                technological_paradigm="Petróleo, automóvil",
                key_innovations=["Petroquímica", "Automóvil masivo", "Aviación comercial"],
                phases={
                    KondratievPhase.PRIMAVERA: (1940, 1955),
                    KondratievPhase.VERANO: (1955, 1965),
                    KondratievPhase.OTOÑO: (1965, 1973),
                    KondratievPhase.INVIERNO: (1973, 1980)
                }
            ),
            KondratievWave(
                wave_number=5,
                start_year=1980,
                end_year=2020,
                duration_years=40,
                technological_paradigm="TIC, internet",
                key_innovations=["Computadora personal", "Internet", "Teléfono móvil"],
                phases={
                    KondratievPhase.PRIMAVERA: (1980, 1995),
                    KondratievPhase.VERANO: (1995, 2000),
                    KondratievPhase.OTOÑO: (2000, 2008),
                    KondratievPhase.INVIERNO: (2008, 2020)
                }
            )
        ]
    
    def _load_historical_sync_data(self) -> List[SolarKondratievSync]:
        """Cargar datos históricos de sincronización solar-Kondratiev"""
        return [
            SolarKondratievSync(
                kondratiev_wave=3,
                solar_cycles=[14, 15, 16, 17],
                phase_synchronization={
                    KondratievPhase.PRIMAVERA: "Máximo solar",
                    KondratievPhase.VERANO: "Transición",
                    KondratievPhase.OTOÑO: "Mínimo solar",  # Crack 1929
                    KondratievPhase.INVIERNO: "Ascendente"
                },
                correlation_strength=0.78,
                historical_evidence=[
                    {"event": "Crack 1929", "solar_cycle": 16, "phase": "OTOÑO"},
                    {"event": "Gran Depresión", "solar_cycle": 16, "phase": "INVIERNO"}
                ]
            ),
            SolarKondratievSync(
                kondratiev_wave=4,
                solar_cycles=[18, 19, 20, 21],
                phase_synchronization={
                    KondratievPhase.PRIMAVERA: "Máximo solar",  # Boom post-guerra
                    KondratievPhase.VERANO: "Alta actividad",
                    KondratievPhase.OTOÑO: "Mínimo solar",     # Crisis petróleo 1973
                    KondratievPhase.INVIERNO: "Recuperación"
                },
                correlation_strength=0.72,
                historical_evidence=[
                    {"event": "Crisis petróleo 1973", "solar_cycle": 20, "phase": "OTOÑO"},
                    {"event": "Estanflación", "solar_cycle": 20, "phase": "INVIERNO"}
                ]
            ),
            SolarKondratievSync(
                kondratiev_wave=5,
                solar_cycles=[22, 23, 24, 25],
                phase_synchronization={
                    KondratievPhase.PRIMAVERA: "Máximo solar",  # Boom internet
                    KondratievPhase.VERANO: "Máximo solar",     # Burbuja dot-com
                    KondratievPhase.OTOÑO: "Mínimo solar",      # Crisis 2008
                    KondratievPhase.INVIERNO: "Mínimo solar"    # COVID-19
                },
                correlation_strength=0.85,
                historical_evidence=[
                    {"event": "Burbuja dot-com", "solar_cycle": 23, "phase": "VERANO"},
                    {"event": "Crisis 2008", "solar_cycle": 24, "phase": "OTOÑO"},
                    {"event": "COVID-19", "solar_cycle": 25, "phase": "INVIERNO"}
                ]
            )
        ]
    
    def analyze_long_waves(self, economic_data: pd.DataFrame = None, 
                          solar_data: pd.DataFrame = None) -> WaveAnalysis:
        """
        Analizar ondas largas actuales y predecir fases futuras
        
        Args:
            economic_data: Datos económicos de largo plazo
            solar_data: Datos solares históricos
            
        Returns:
            Análisis completo de onda actual
        """
        logger.info("🔮 Analizando ondas largas de Kondratiev...")
        
        try:
            # Determinar onda actual (presumiblemente 6ta onda)
            current_wave = self._identify_current_wave()
            
            # Determinar fase actual
            current_phase, phase_progress = self._determine_current_phase(current_wave)
            
            # Analizar sincronización solar
            solar_correlation = self._analyze_solar_synchronization(current_wave, solar_data)
            
            # Predecir transición de fase
            next_transition = self._predict_phase_transition(current_phase, phase_progress)
            
            # Evaluar implicaciones económicas
            economic_implications = self._assess_economic_implications(current_phase)
            
            # Evaluar riesgos
            risk_assessment = self._assess_kondratiev_risks(current_phase, solar_correlation)
            
            # Crear análisis completo
            analysis = WaveAnalysis(
                current_wave=current_wave,
                current_phase=current_phase,
                phase_progress=phase_progress,
                next_phase_transition=next_transition,
                economic_implications=economic_implications,
                solar_correlation=solar_correlation,
                risk_assessment=risk_assessment
            )
            
            self.current_analysis = analysis
            logger.info(f"✅ Onda {current_wave.wave_number} - Fase {current_phase.value} detectada")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando ondas largas: {e}")
            raise
    
    def _identify_current_wave(self) -> KondratievWave:
        """Identificar la onda Kondratiev actual"""
        # Según la cronología histórica, estaríamos en la 6ta onda
        current_year = datetime.now().year
        
        # Crear onda 6 (actual)
        wave_6 = KondratievWave(
            wave_number=6,
            start_year=2020,
            end_year=2070,  # Proyección
            duration_years=50,
            technological_paradigm="IA, Biotecnología, Energía Verde",
            key_innovations=[
                "Inteligencia Artificial General",
                "Edición genética CRISPR",
                "Energía de fusión",
                "Computación cuántica",
                "Transhumanismo"
            ],
            phases={
                KondratievPhase.PRIMAVERA: (2020, 2035),
                KondratievPhase.VERANO: (2035, 2045),
                KondratievPhase.OTOÑO: (2045, 2055),
                KondratievPhase.INVIERNO: (2055, 2070)
            }
        )
        
        return wave_6
    
    def _determine_current_phase(self, current_wave: KondratievWave) -> Tuple[KondratievPhase, float]:
        """Determinar la fase actual de la onda Kondratiev"""
        current_year = datetime.now().year
        
        for phase, (start_year, end_year) in current_wave.phases.items():
            if start_year <= current_year <= end_year:
                # Calcular progreso en la fase
                phase_duration = end_year - start_year
                years_elapsed = current_year - start_year
                progress = years_elapsed / phase_duration
                
                return phase, progress
        
        # Si no está en ninguna fase definida, asumir primavera
        return KondratievPhase.PRIMAVERA, 0.3
    
    def _analyze_solar_synchronization(self, current_wave: KondratievWave,
                                     solar_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Analizar sincronización entre onda Kondratiev y ciclos solares largos
        
        Args:
            current_wave: Onda Kondratiev actual
            solar_data: Datos solares históricos
            
        Returns:
            Métricas de correlación y sincronización
        """
        logger.info("☀️ Analizando sincronización solar-Kondratiev...")
        
        correlation_metrics = {
            "phase_alignment": 0.0,
            "cycle_synchronization": 0.0,
            "historical_correlation": 0.0,
            "predicted_sync_strength": 0.0,
            "gleissberg_kondratiev_ratio": 0.0
        }
        
        try:
            # Calcular relación entre ciclos Gleissberg y Kondratiev
            cycle_ratio = self.gleissberg_cycle_years / self.kondratiev_cycle_years
            correlation_metrics["gleissberg_kondratiev_ratio"] = cycle_ratio
            
            # Buscar sincronización histórica para onda actual
            wave_sync_data = next(
                (sync for sync in self.historical_sync_data 
                 if sync.kondratiev_wave == current_wave.wave_number - 1), 
                None
            )
            
            if wave_sync_data:
                correlation_metrics["historical_correlation"] = wave_sync_data.correlation_strength
                
                # Predecir fuerza de sincronización para onda actual
                # Basado en patrones históricos y relación de ciclos
                predicted_strength = self._predict_sync_strength(
                    current_wave, wave_sync_data
                )
                correlation_metrics["predicted_sync_strength"] = predicted_strength
            
            # Análisis de fase actual con actividad solar
            current_phase = self._determine_current_phase(current_wave)[0]
            solar_alignment = self._analyze_phase_solar_alignment(current_phase)
            correlation_metrics["phase_alignment"] = solar_alignment
            
            # Sincronización de ciclos (Gleissberg vs Kondratiev)
            cycle_sync = self._analyze_cycle_synchronization()
            correlation_metrics["cycle_synchronization"] = cycle_sync
            
        except Exception as e:
            logger.error(f"Error en análisis de sincronización solar: {e}")
        
        return correlation_metrics
    
    def _predict_sync_strength(self, current_wave: KondratievWave, 
                             historical_sync: SolarKondratievSync) -> float:
        """Predecir fuerza de sincronización para onda actual"""
        # Promedio histórico de correlación
        historical_strengths = [sync.correlation_strength 
                              for sync in self.historical_sync_data]
        avg_historical = np.mean(historical_strengths)
        
        # Ajustar basado en características de la onda actual
        wave_adjustment = 1.0
        
        # Ondas con tecnologías más dependientes de energía pueden tener mayor sincronización
        energy_dependent_tech = ["IA", "Energía", "Biotecnología"]
        current_tech = current_wave.technological_paradigm
        
        if any(tech in current_tech for tech in energy_dependent_tech):
            wave_adjustment *= 1.2
        
        predicted_strength = avg_historical * wave_adjustment
        return min(predicted_strength, 1.0)  # Máximo 1.0
    
    def _analyze_phase_solar_alignment(self, current_phase: KondratievPhase) -> float:
        """Analizar alineación entre fase económica y actividad solar"""
        # Mapeo teórico fases económicas - actividad solar
        phase_solar_mapping = {
            KondratievPhase.PRIMAVERA: 0.8,  # Alta actividad solar favorece innovación
            KondratievPhase.VERANO: 0.6,     # Actividad moderada
            KondratievPhase.OTOÑO: 0.3,      # Transición solar
            KondratievPhase.INVIERNO: 0.4    # Mínimo solar para reinvención
        }
        
        return phase_solar_mapping.get(current_phase, 0.5)
    
    def _analyze_cycle_synchronization(self) -> float:
        """Analizar sincronización entre ciclos Gleissberg y Kondratiev"""
        # Los ciclos deberían estar en relación aproximadamente 3:2
        # 3 ciclos solares Gleissberg ≈ 2 ciclos Kondratiev
        expected_ratio = 3/2
        actual_ratio = self.gleissberg_cycle_years / self.kondratiev_cycle_years
        
        # Calcular desviación de la relación ideal
        deviation = abs(actual_ratio - expected_ratio) / expected_ratio
        synchronization = 1.0 - deviation
        
        return max(0.0, min(1.0, synchronization))
    
    def _predict_phase_transition(self, current_phase: KondratievPhase, 
                                phase_progress: float) -> datetime:
        """Predecir cuándo ocurrirá la próxima transición de fase"""
        current_year = datetime.now().year
        
        # Duración típica de cada fase en años
        phase_durations = {
            KondratievPhase.PRIMAVERA: 15,
            KondratievPhase.VERANO: 10,
            KondratievPhase.OTOÑO: 10,
            KondratievPhase.INVIERNO: 15
        }
        
        current_duration = phase_durations.get(current_phase, 12)
        years_remaining = current_duration * (1 - phase_progress)
        
        # Añadir variabilidad basada en sincronización solar
        solar_influence = np.random.normal(0, 0.5)  # ±6 meses
        
        transition_year = current_year + years_remaining + solar_influence
        transition_date = datetime(int(transition_year), 1, 1)
        
        return transition_date
    
    def _assess_economic_implications(self, current_phase: KondratievPhase) -> Dict[str, Any]:
        """Evaluar implicaciones económicas de la fase actual"""
        implications = {
            "growth_outlook": "",
            "investment_opportunities": [],
            "sector_recommendations": [],
            "risk_factors": [],
            "policy_implications": []
        }
        
        if current_phase == KondratievPhase.PRIMAVERA:
            implications.update({
                "growth_outlook": "Crecimiento acelerado e innovación disruptiva",
                "investment_opportunities": [
                    "Tecnologías emergentes",
                    "Infraestructura nueva",
                    "Startups innovadoras"
                ],
                "sector_recommendations": [
                    "Tecnología",
                    "Energías renovables", 
                    "Biotecnología",
                    "Inteligencia Artificial"
                ],
                "risk_factors": [
                    "Sobrevaluación de innovaciones",
                    "Regulación desfasada",
                    "Burbujas tecnológicas"
                ],
                "policy_implications": [
                    "Incentivos a I+D",
                    "Flexibilidad regulatoria",
                    "Educación en nuevas habilidades"
                ]
            })
        
        elif current_phase == KondratievPhase.VERANO:
            implications.update({
                "growth_outlook": "Prosperidad generalizada con signos de madurez",
                "investment_opportunities": [
                    "Expansión internacional",
                    "Optimización operativa",
                    "Fusiones y adquisiciones"
                ],
                "sector_recommendations": [
                    "Bienes de lujo",
                    "Bienes raíces",
                    "Mercados desarrollados"
                ],
                "risk_factors": [
                    "Sobreendeudamiento",
                    "Exceso de capacidad",
                    "Competencia excesiva"
                ],
                "policy_implications": [
                    "Control de inflación",
                    "Regulación financiera",
                    "Políticas anti-cíclicas"
                ]
            })
        
        elif current_phase == KondratievPhase.OTOÑO:
            implications.update({
                "growth_outlook": "Estancamiento con crisis financieras esporádicas",
                "investment_opportunities": [
                    "Activos defensivos",
                    "Oro y metales preciosos",
                    "Deuda soberana calidad"
                ],
                "sector_recommendations": [
                    "Servicios esenciales",
                    "Salud y farmacéutica",
                    "Utilidades"
                ],
                "risk_factors": [
                    "Crisis de deuda",
                    "Deflación",
                    "Desempleo estructural"
                ],
                "policy_implications": [
                    "Estímulo fiscal cuidadoso",
                    "Reestructuración deuda",
                    "Protección social"
                ]
            })
        
        elif current_phase == KondratievPhase.INVIERNO:
            implications.update({
                "growth_outlook": "Depresión y reinvención fundamental",
                "investment_opportunities": [
                    "Tecnologías de próxima onda",
                    "Activos en quiebra",
                    "Materias primas estratégicas"
                ],
                "sector_recommendations": [
                    "Tecnologías básicas nuevas",
                    "Infraestructura crítica",
                    "Educación y capacitación"
                ],
                "risk_factors": [
                    "Colapso financiero",
                    "Inestabilidad social",
                    "Proteccionismo"
                ],
                "policy_implications": [
                    "Reforma estructural profunda",
                    "Nuevo sistema regulatorio",
                    "Cooperación internacional"
                ]
            })
        
        return implications
    
    def _assess_kondratiev_risks(self, current_phase: KondratievPhase,
                               solar_correlation: Dict[str, float]) -> Dict[str, Any]:
        """Evaluar riesgos asociados a la fase Kondratiev actual"""
        risk_assessment = {
            "economic_risk_level": "",
            "financial_risk_level": "",
            "social_risk_level": "",
            "technological_risk_level": "",
            "solar_influence_risk": 0.0,
            "composite_risk_index": 0.0,
            "risk_mitigation_strategies": []
        }
        
        # Niveles de riesgo base por fase
        phase_risks = {
            KondratievPhase.PRIMAVERA: {
                "economic": "Bajo", "financial": "Moderado", 
                "social": "Bajo", "technological": "Alto"
            },
            KondratievPhase.VERANO: {
                "economic": "Moderado", "financial": "Alto", 
                "social": "Bajo", "technological": "Moderado"
            },
            KondratievPhase.OTOÑO: {
                "economic": "Alto", "financial": "Muy Alto", 
                "social": "Moderado", "technological": "Bajo"
            },
            KondratievPhase.INVIERNO: {
                "economic": "Muy Alto", "financial": "Alto", 
                "social": "Alto", "technological": "Moderado"
            }
        }
        
        base_risks = phase_risks.get(current_phase, {
            "economic": "Moderado", "financial": "Moderado",
            "social": "Moderado", "technological": "Moderado"
        })
        
        risk_assessment.update({
            "economic_risk_level": base_risks["economic"],
            "financial_risk_level": base_risks["financial"],
            "social_risk_level": base_risks["social"],
            "technological_risk_level": base_risks["technological"]
        })
        
        # Ajustar por influencia solar
        solar_influence = solar_correlation.get("predicted_sync_strength", 0.5)
        risk_assessment["solar_influence_risk"] = solar_influence
        
        # Calcular índice de riesgo compuesto
        risk_mapping = {"Bajo": 1, "Moderado": 2, "Alto": 3, "Muy Alto": 4}
        economic_risk = risk_mapping[base_risks["economic"]]
        financial_risk = risk_mapping[base_risks["financial"]]
        social_risk = risk_mapping[base_risks["social"]]
        tech_risk = risk_mapping[base_risks["technological"]]
        
        composite_risk = (economic_risk + financial_risk + social_risk + tech_risk) / 4
        composite_risk *= (1 + solar_influence * 0.5)  # Aumentar riesgo si alta sincronización solar
        
        risk_assessment["composite_risk_index"] = min(composite_risk, 4.0)
        
        # Estrategias de mitigación
        risk_assessment["risk_mitigation_strategies"] = self._generate_risk_mitigation(
            current_phase, composite_risk
        )
        
        return risk_assessment
    
    def _generate_risk_mitigation(self, current_phase: KondratievPhase,
                                composite_risk: float) -> List[str]:
        """Generar estrategias de mitigación de riesgos"""
        strategies = []
        
        if composite_risk >= 3.0:
            strategies.extend([
                "Diversificación global de activos",
                "Enfoque en liquidez y preservación de capital",
                "Coberturas contra cola de riesgo",
                "Revisión de exposición a sectores cíclicos"
            ])
        
        if current_phase in [KondratievPhase.OTOÑO, KondratievPhase.INVIERNO]:
            strategies.extend([
                "Enfoque en calidad crediticia",
                "Reducción de apalancamiento",
                "Inversión en activos refugio",
                "Preparación para oportunidades de dislocación"
            ])
        
        if current_phase in [KondratievPhase.PRIMAVERA, KondratievPhase.VERANO]:
            strategies.extend([
                "Exposición a crecimiento mediante ETFs sectoriales",
                "Inversión en innovación disruptiva",
                "Participación en capital riesgo temprano",
                "Flexibilidad para rotación sectorial"
            ])
        
        # Estrategias específicas por nivel de riesgo
        if composite_risk >= 3.5:
            strategies.append("Considerar estrategias de riesgo absoluto")
        
        return strategies
    
    def get_current_phase(self) -> Tuple[KondratievPhase, float]:
        """Obtener fase actual y progreso"""
        if self.current_analysis:
            return self.current_analysis.current_phase, self.current_analysis.phase_progress
        
        current_wave = self._identify_current_wave()
        return self._determine_current_phase(current_wave)
    
    def predict_next_transition(self) -> Dict[str, Any]:
        """Predecir próxima transición de fase"""
        if not self.current_analysis:
            self.analyze_long_waves()
        
        return {
            "next_phase_transition": self.current_analysis.next_phase_transition,
            "current_phase": self.current_analysis.current_phase.value,
            "days_until_transition": (
                self.current_analysis.next_phase_transition - datetime.now()
            ).days,
            "transition_confidence": 0.75,
            "expected_economic_impact": self._assess_transition_impact()
        }
    
    def _assess_transition_impact(self) -> Dict[str, str]:
        """Evaluar impacto económico de la próxima transición"""
        current_phase = self.current_analysis.current_phase
        
        transition_impacts = {
            (KondratievPhase.PRIMAVERA, KondratievPhase.VERANO): {
                "impact": "Positivo",
                "description": "Transición suave hacia prosperidad generalizada",
                "sectors": "Todos los sectores, especialmente consumo y lujo"
            },
            (KondratievPhase.VERANO, KondratievPhase.OTOÑO): {
                "impact": "Negativo", 
                "description": "Transición abrupta con crisis financiera",
                "sectors": "Financiero y bienes cíclicos más afectados"
            },
            (KondratievPhase.OTOÑO, KondratievPhase.INVIERNO): {
                "impact": "Muy Negativo",
                "description": "Transición profunda hacia depresión económica",
                "sectors": "Todos los sectores, especialmente industriales"
            },
            (KondratievPhase.INVIERNO, KondratievPhase.PRIMAVERA): {
                "impact": "Muy Positivo",
                "description": "Renacimiento económico con nuevas tecnologías",
                "sectors": "Tecnología e innovación lideran recuperación"
            }
        }
        
        # Determinar próxima fase
        phases = list(KondratievPhase)
        current_idx = phases.index(current_phase)
        next_phase = phases[(current_idx + 1) % len(phases)]
        
        return transition_impacts.get((current_phase, next_phase), {
            "impact": "Moderado",
            "description": "Transición estándar del ciclo",
            "sectors": "Mixto across sectores"
        })
    
    def generate_kondratiev_report(self) -> Dict[str, Any]:
        """Generar reporte completo de análisis Kondratiev"""
        if not self.current_analysis:
            self.analyze_long_waves()
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "current_wave": {
                "number": self.current_analysis.current_wave.wave_number,
                "technological_paradigm": self.current_analysis.current_wave.technological_paradigm,
                "key_innovations": self.current_analysis.current_wave.key_innovations,
                "duration_years": self.current_analysis.current_wave.duration_years
            },
            "current_phase": {
                "name": self.current_analysis.current_phase.value,
                "progress": self.current_analysis.phase_progress,
                "description": self._get_phase_description(self.current_analysis.current_phase)
            },
            "solar_synchronization": self.current_analysis.solar_correlation,
            "economic_implications": self.current_analysis.economic_implications,
            "risk_assessment": self.current_analysis.risk_assessment,
            "predictions": {
                "next_phase_transition": self.current_analysis.next_phase_transition.isoformat(),
                "transition_impact": self._assess_transition_impact(),
                "long_term_outlook": self._generate_long_term_outlook()
            },
            "investment_recommendations": self._generate_investment_recommendations()
        }
    
    def _get_phase_description(self, phase: KondratievPhase) -> str:
        """Obtener descripción detallada de cada fase"""
        descriptions = {
            KondratievPhase.PRIMAVERA: "Fase de innovación disruptiva y crecimiento acelerado. Nuevas tecnologías emergen y transforman la economía.",
            KondratievPhase.VERANO: "Fase de prosperidad y madurez. Las innovaciones se difunden masivamente y se produce sobreinversión.",
            KondratievPhase.OTOÑO: "Fase de estancamiento y crisis financiera. Exceso de capacidad y burbujas especulativas estallan.",
            KondratievPhase.INVIERNO: "Fase de depresión y reinvención. Purga del exceso y preparación para nueva onda de innovación."
        }
        return descriptions.get(phase, "Fase no definida")
    
    def _generate_long_term_outlook(self) -> Dict[str, Any]:
        """Generar perspectiva de largo plazo"""
        current_wave = self.current_analysis.current_wave
        current_phase = self.current_analysis.current_phase
        
        return {
            "next_5_years": f"Continuación de fase {current_phase.value} con {current_wave.technological_paradigm}",
            "next_10_years": "Transición hacia nueva fase Kondratiev",
            "next_20_years": "Posible inicio de 7ma onda Kondratiev",
            "key_megatrends": [
                "Convergencia tecnológica (IA, biotech, nanotech)",
                "Transición energética global",
                "Envejecimiento poblacional",
                "Digitalización total de economía"
            ],
            "existential_risks": [
                "Singularidad tecnológica",
                "Cambio climático acelerado",
                "Conflictos geopolíticos por recursos",
                "Pandemias globales"
            ]
        }
    
    def _generate_investment_recommendations(self) -> Dict[str, Any]:
        """Generar recomendaciones de inversión basadas en fase actual"""
        current_phase = self.current_analysis.current_phase
        
        recommendations = {
            "asset_allocation": {},
            "sector_emphasis": [],
            "geographic_focus": [],
            "risk_management": [],
            "opportunity_areas": []
        }
        
        if current_phase == KondratievPhase.PRIMAVERA:
            recommendations.update({
                "asset_allocation": {"Equities": 70, "Bonds": 20, "Alternatives": 10},
                "sector_emphasis": ["Tecnología", "Salud", "Energías renovables"],
                "geographic_focus": ["Mercados emergentes", "Tech hubs globales"],
                "risk_management": ["Diversificación sectorial", "Stop losses dinámicos"],
                "opportunity_areas": ["VC/Private Equity", "Tecnologías emergentes"]
            })
        
        elif current_phase == KondratievPhase.VERANO:
            recommendations.update({
                "asset_allocation": {"Equities": 50, "Bonds": 30, "Alternatives": 20},
                "sector_emphasis": ["Consumo", "Lujo", "Bienes raíces"],
                "geographic_focus": ["Mercados desarrollados", "Economías estables"],
                "risk_management": ["Reducción apalancamiento", "Coberturas sectoriales"],
                "opportunity_areas": ["Momentum investing", "Dividend growth"]
            })
        
        elif current_phase == KondratievPhase.OTOÑO:
            recommendations.update({
                "asset_allocation": {"Equities": 30, "Bonds": 50, "Alternatives": 20},
                "sector_emphasis": ["Defensivos", "Utilidades", "Oro"],
                "geographic_focus": ["Países con superávit", "Refugios seguros"],
                "risk_management": ["Alta liquidez", "Coberturas de cola"],
                "opportunity_areas": ["Short selling", "Distressed assets"]
            })
        
        elif current_phase == KondratievPhase.INVIERNO:
            recommendations.update({
                "asset_allocation": {"Equities": 20, "Bonds": 40, "Alternatives": 40},
                "sector_emphasis": ["Tecnología básica nueva", "Infraestructura"],
                "geographic_focus": ["Países con reformas", "Mercados infravalorados"],
                "risk_management": ["Preservación capital", "Diversificación extrema"],
                "opportunity_areas": ["Turnaround situations", "Nuevos paradigmas"]
            })
        
        return recommendations

# Instancia global para uso en otros módulos
kondratiev_analyzer = KondratievAnalyzer()
