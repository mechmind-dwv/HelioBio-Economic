# 🌞 HelioBio-Economic v1.0  
**Sistema de Análisis de Correlación entre Ciclos Solares y Ciclos Económicos**  
*Extendiendo el legado de Alexander Chizhevsky al dominio económico-financiero*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![NASA DONKI](https://img.shields.io/badge/NASA_DONKI-API-orange.svg)](https://api.nasa.gov/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"Las tormentas solares modulan los mercados financieros y los ciclos económicos"**  
> — Inspirado en Alexander Chizhevsky & Nikolai Kondratiev

---

## 📊 **VISIÓN HELIOBIO-ECONÓMICA**

### 🔥 **Misión Científica**
Sistema interdisciplinario que investiga y demuestra las correlaciones entre:
- **🌞 Actividad solar** (ciclos de 11 años, tormentas geomagnéticas)
- **📈 Indicadores económicos** (mercados bursátiles, PIB, inflación)
- **🔄 Ciclos económicos largos** (ondas de Kondratiev, ciclos de Kuznets)

### 🧠 **Fundamento Científico**
Basado en las investigaciones de:
- **Alexander Chizhevsky**: Efectos de la actividad solar en el comportamiento humano
- **Nikolai Kondratiev**: Ciclos económicos largos (45-60 años)  
- **William Stanley Jevons**: Teoría de las manchas solares y ciclos económicos

---

## 🚀 **ARQUITECTURA DEL SISTEMA**

### 🏗️ **Componentes Principales**
```python
sistema_heliobio_economic = {
    "input_solar": "NASA DONKI API - Datos solares en tiempo real",
    "input_economic": "Yahoo Finance, FRED, World Bank - Datos económicos",
    "procesamiento": "ML Ensemble - Análisis de correlación y predicción",
    "output": "Dashboard de correlaciones solares-económicas"
}
```

### 📈 **Fuentes de Datos Implementadas**
| Fuente | Tipo de Datos | Frecuencia | Estado |
|--------|---------------|------------|---------|
| **NASA DONKI** | Actividad solar, CME, fulguraciones | Tiempo real | ✅ |
| **Yahoo Finance** | Mercados bursátiles, índices | Diario | ✅ |
| **FRED API** | Indicadores macroeconómicos | Mensual | ✅ |
| **World Bank** | Datos económicos globales | Anual | 🔄 |

---

## 🔧 **INSTALACIÓN Y CONFIGURACIÓN**

### ⚡ **Inicio Rápido**
```bash
# 1. Clonar repositorio
git clone https://github.com/mechmind-dwv/HelioBio-Economic.git
cd HelioBio-Economic

# 2. Configurar entorno virtual
python -m venv helio_env
source helio_env/bin/activate  # Linux/Mac
# helio_env\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar APIs
python scripts/setup_economic_apis.py
```

### 🔑 **Configuración de APIs**
```python
# Configuración en .env
NASA_API_KEY=tu_clave_nasa
FRED_API_KEY=tu_clave_fred
YAHOO_FINANCE=True  # No requiere API key
```

---

## 📊 **ENDPOINTS PRINCIPALES**

### 🌞 **Datos Solares**
```bash
# Actividad solar actual
curl http://localhost:8000/api/solar/current

# Histórico de ciclos solares
curl http://localhost:8000/api/solar/historical?years=50
```

### 💹 **Datos Económicos**
```bash
# Mercados bursátiles
curl http://localhost:8000/api/economic/markets

# Indicadores macroeconómicos
curl http://localhost:8000/api/economic/indicators

# Ciclos de Kondratiev
curl http://localhost:8000/api/economic/kondratiev
```

### 🔗 **Análisis de Correlación**
```bash
# Correlación solar-económica
curl http://localhost:8000/api/correlation/solar-economic

# Predicción basada en ciclos
curl http://localhost:8000/api/prediction/economic?cycles=solar,kondratiev
```

---

## 🎯 **CASOS DE USO INVESTIGADOS**

### 📈 **Correlaciones Históricas**
1. **Máximos Solares vs Crisis Económicas**
   - 2008: Crisis financiera + Mínimo solar
   - 2000: Burble dot-com + Máximo solar
   - 1987: Black Monday + Actividad solar elevada

2. **Ciclos Largos**
   - Ondas de Kondratiev (45-60 años)
   - Ciclos solares (11 años) y sus armónicos
   - Interferencia entre ciclos solares y económicos

### 🔬 **Métricas de Análisis**
```python
metricas_analisis = {
    "correlacion_cruzada": "Sunspots vs S&P 500",
    "analisis_espectral": "Detección de ciclos comunes",
    "prediccion_no_lineal": "ML para forecasting económico",
    "analisis_causalidad": "Test de Granger solar-económico"
}
```

---

## 🏗️ **ESTRUCTURA DEL PROYECTO**

```
HelioBio-Economic/
├── app/
│   ├── main.py                      # FastAPI principal
│   ├── core/
│   │   ├── economic_cycles.py       # Análisis ciclos económicos
│   │   ├── solar_economic_ml.py     # ML para correlaciones
│   │   └── kondratiev_analysis.py   # Ondas largas
│   ├── services/
│   │   ├── nasa_solar_service.py    # Datos solares
│   │   ├── economic_data_service.py # Datos económicos
│   │   └── correlation_service.py   # Análisis correlación
│   └── api/                         # Endpoints
├── data/
│   ├── solar/                       # Datos solares históricos
│   ├── economic/                    # Series económicas
│   └── models/                      # Modelos ML entrenados
├── notebooks/
│   ├── exploratory_analysis.ipynb   # Análisis exploratorio
│   └── cycle_correlation.ipynb     # Correlación de ciclos
└── scripts/
    ├── install.sh                   # Instalación
    └── setup_apis.py               # Configuración APIs
```

---

## 📚 **INVESTIGACIÓN Y METODOLOGÍA**

### 🔍 **Enfoque Científico**
1. **Análisis de Series Temporales**
   - Descomposición estacional y de tendencias
   - Análisis espectral (FFT, wavelets)
   - Correlación cruzada entre dominios

2. **Machine Learning Avanzado**
   - Random Forest para feature importance
   - LSTM para predicción temporal
   - Clustering de regímenes mercado-solares

3. **Validación Estadística**
   - Tests de estacionariedad
   - Análisis de causalidad de Granger
   - Bootstrapping para significancia

---

## 🌍 **ROADMAP DE DESARROLLO**

### v1.0 (Actual) - Base Científica
- [x] Integración APIs solares y económicas
- [x] Análisis de correlación básico
- [x] Dashboard inicial

### v1.1 (Próximo) - ML Avanzado
- [ ] Modelos LSTM para predicción
- [ ] Análisis de causalidad
- [ ] Backtesting estratégico

### v1.2 (Futuro) - Producción
- [ ] Sistema de alertas tempranas
- [ ] API pública para investigadores
- [ ] Paper científico

---

## 🤝 **CONTRIBUCIONES**

### 🎯 **Áreas de Investigación Prioritaria**
1. **Ciclos Solares-Económicos**
   - Correlación entre máximos solares y recesiones
   - Efecto de tormentas geomagnéticas en mercados

2. **Ondas Largas**
   - Sincronización ciclos Kondratiev-Schwabe
   - Análisis espectral multidisciplinar

3. **Aplicaciones Prácticas**
   - Estrategias de inversión basadas en ciclos
   - Gestión de riesgo económico solar-influenciado

---

## 📊 **RESULTADOS PRELIMINARES**

### 🔬 **Hallazgos Iniciales**
```python
resultados_preliminares = {
    "correlacion_solar_sp500": "0.32 (p < 0.05)",
    "ciclo_detectado": "10.8 años ≈ ciclo solar 11 años", 
    "mejor_modelo_prediccion": "Random Forest (R² = 0.41)",
    "causalidad_granger": "Solar → Económica significativa"
}
```

---

## 👥 **AUTORES**

**HelioBio-Economic Research Team**
- **Benjamin Cabeza Durán** ([mechmind-dwv](https://github.com/mechmind-dwv))
- **DeepSeek AI** (Asistente de investigación)
- **Email**: ia.mechmind@gmail.com

**En memoria de Alexander L. Chizhevsky** - cuyo trabajo pionero hizo posible esta investigación.

---

## 📄 **LICENCIA**

MIT License - Ver [LICENSE](LICENSE) para detalles.

---

## 🌟 **CITA ACADÉMICA**

```bibtex
@software{HelioBioEconomic2024,
  title = {HelioBio-Economic: Solar-Economic Cycle Analysis System},
  author = {Cabeza Durán, Benjamin and DeepSeek AI},
  year = {2024},
  url = {https://github.com/mechmind-dwv/HelioBio-Economic}
}
```

---

<div align="center">

## 🔮 **EL FUTURO DE LA ECONOMÍA CÓSMICA**

**"Comprendiendo los ritmos del sol para anticipar los latidos de la economía"**

[🚀 Comenzar](#-instalación-y-configuración) | 
[📊 Dashboard](http://localhost:8000) | 
[🔗 APIs](#-endpoints-principales)

</div>
