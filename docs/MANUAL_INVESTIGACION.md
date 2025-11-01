# 🛠️ Manual del Desarrollador - HelioBio-Economic v1.0

## 📖 Índice
1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Configuración del Entorno](#configuración-del-entorno)
4. [Guía de Desarrollo](#guía-de-desarrollo)
5. [APIs y Servicios](#apis-y-servicios)
6. [Base de Datos y Cache](#base-de-datos-y-cache)
7. [Testing y Calidad](#testing-y-calidad)
8. [Despliegue](#despliegue)
9. [Contribución](#contribución)

## 🏗️ Arquitectura del Sistema

### Diagrama de Arquitectura
```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Frontend      │    │   Backend        │    │   Servicios      │
│   Dashboard     │    │   FastAPI        │    │   Externos       │
│   HTML/JS/CSS   │◄──►│   Python 3.8+    │◄──►│   NASA, FRED,    │
│   Chart.js      │    │   Uvicorn        │    │   Yahoo Finance  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                              │
                      ┌──────────────────┐
                      │   Datos & Cache  │
                      │   Pandas         │
                      │   Redis*         │
                      │   Archivos       │
                      └──────────────────┘
```

### Componentes Principales

**1. Frontend (app/static/)**
- Dashboard SPA con vanilla JavaScript
- Chart.js para visualizaciones
- Diseño responsive con CSS Grid/Flexbox

**2. Backend API (app/api/)**
- FastAPI con documentación automática
- 22 endpoints organizados en routers
- Validación con Pydantic models

**3. Servicios de Datos (app/services/)**
- NASA Solar Service (DONKI API)
- Economic Data Service (Yahoo Finance, FRED)
- Correlation Service (análisis avanzado)

**4. Núcleo Analítico (app/core/)**
- Economic Cycles (ciclos económicos)
- Solar Economic ML (machine learning)
- Kondratiev Analysis (ondas largas)

**5. Gestión de Datos (data/)**
- Datos históricos solares y económicos
- Modelos ML entrenados
- Sistema de cache distribuido

## 📁 Estructura del Proyecto

```
HelioBio-Economic/
├── app/                          # Código de la aplicación
│   ├── main.py                   # Punto de entrada FastAPI
│   ├── core/                     # Lógica de negocio
│   │   ├── economic_cycles.py    # Análisis ciclos económicos
│   │   ├── solar_economic_ml.py  # Modelos ML
│   │   └── kondratiev_analysis.py # Ondas largas
│   ├── services/                 # Servicios de datos
│   │   ├── nasa_solar_service.py # Datos solares NASA
│   │   ├── economic_data_service.py # Datos económicos
│   │   └── correlation_service.py # Análisis correlación
│   ├── api/                      # Endpoints FastAPI
│   │   ├── routers.py            # Configuración routers
│   │   ├── endpoints/            # Grupos de endpoints
│   │   └── models/               # Modelos Pydantic
│   └── static/                   # Frontend estático
│       ├── index.html            # Dashboard principal
│       ├── app.js                # Lógica frontend
│       └── styles.css            # Estilos
├── data/                         # Datos y modelos
│   ├── solar/                    # Datos solares históricos
│   ├── economic/                 # Series económicas
│   ├── models/                   # Modelos ML entrenados
│   └── cache/                    # Sistema de cache
├── scripts/                      # Scripts de utilidad
│   ├── install.sh                # Instalación automática
│   └── setup_apis.py             # Configuración APIs
├── notebooks/                    # Análisis exploratorios
│   ├── exploratory_analysis.ipynb
│   └── cycle_correlation.ipynb
├── tests/                        # Suite de tests
├── docs/                         # Documentación
├── requirements.txt              # Dependencias Python
├── .env.example                  # Variables de entorno
└── README.md                     # Documentación principal
```

## ⚙️ Configuración del Entorno

### Requisitos del Sistema de Desarrollo
- **Python**: 3.8, 3.9, 3.10, 3.11
- **RAM**: 8GB mínimo, 16GB recomendado
- **Storage**: 10GB para datos históricos
- **OS**: Linux, macOS, Windows (WSL2 recomendado para Windows)

### Configuración Inicial

**1. Clonar y Configurar**
```bash
git clone https://github.com/mechmind-dwv/HelioBio-Economic.git
cd HelioBio-Economic

# Crear entorno virtual
python -m venv helio_env
source helio_env/bin/activate  # Linux/macOS
# helio_env\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

**2. Configurar Variables de Entorno**
```bash
cp .env.example .env
# Editar .env con tus claves API
```

**Variables de Entorno Requeridas:**
```env
# NASA API (obligatoria)
NASA_API_KEY=tu_clave_nasa

# APIs Económicas (opcionales pero recomendadas)
FRED_API_KEY=tu_clave_fred
ALPHA_VANTAGE_KEY=tu_clave_alpha_vantage

# Configuración de la Aplicación
DEBUG=True
LOG_LEVEL=INFO
CACHE_TTL=3600
```

**3. Verificar Instalación**
```bash
# Ejecutar tests básicos
python -m pytest tests/ -v

# Iniciar servidor de desarrollo
python app/main.py
```

## 💻 Guía de Desarrollo

### Estructura de un Nuevo Endpoint

**1. Crear Modelo de Respuesta (app/api/models/responses.py)**
```python
class NewAnalysisResponse(StandardResponse):
    data: Optional[Dict[str, Any]] = Field(None, description="Resultados del nuevo análisis")
```

**2. Crear Endpoint (app/api/endpoints/nuevo_analisis.py)**
```python
router = APIRouter()

@router.get("/nuevo-analisis", response_model=NewAnalysisResponse)
async def get_new_analysis(parametro: str = Query(...)):
    try:
        # Lógica del endpoint
        resultado = await some_service.analyze(parametro)
        
        return NewAnalysisResponse(
            success=True,
            message="Análisis completado",
            data=resultado
        )
    except Exception as e:
        logger.error(f"Error en nuevo análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**3. Registrar Router (app/api/routers.py)**
```python
from app.api.endpoints import nuevo_analisis

api_router.include_router(
    nuevo_analisis.router, 
    prefix="/nuevo-analisis", 
    tags=["Nuevo Análisis"]
)
```

### Convenciones de Código

**Nomenclatura:**
- **Variables**: `snake_case` (ej: `solar_activity`)
- **Clases**: `PascalCase` (ej: `SolarActivityAnalyzer`)
- **Constantes**: `UPPER_SNAKE_CASE` (ej: `MAX_RETRIES`)
- **Métodos**: `snake_case` (ej: `calculate_correlation()`)

**Documentación:**
```python
def calculate_correlation(series1: pd.Series, series2: pd.Series) -> CorrelationResult:
    """
    Calcula la correlación entre dos series temporales
    
    Args:
        series1: Primera serie temporal
        series2: Segunda serie temporal
        
    Returns:
        CorrelationResult: Resultado del análisis de correlación
        
    Raises:
        ValueError: Si las series tienen longitudes diferentes
    """
    # Implementación...
```

**Logging:**
```python
import logging

logger = logging.getLogger(__name__)

def some_function():
    try:
        logger.info("Iniciando procesamiento...")
        # Código...
        logger.info("Procesamiento completado")
    except Exception as e:
        logger.error(f"Error en procesamiento: {e}")
        raise
```

## 🔌 APIs y Servicios

### Servicio NASA Solar

**Clase Principal**: `NASASolarService`
**Responsabilidad**: Obtener datos solares de NASA DONKI API

**Métodos Principales:**
```python
async def get_current_solar_activity() -> SolarActivitySummary
async def get_solar_flares(days: int = 7) -> List[SolarFlare]
async def get_historical_solar_data(years: int = 50) -> pd.DataFrame
```

**Ejemplo de Uso:**
```python
from app.services.nasa_solar_service import nasa_solar_service

# Obtener actividad solar actual
activity = await nasa_solar_service.get_current_solar_activity()
print(f"Manchas solares: {activity.sunspot_number}")
```

### Servicio de Datos Económicos

**Clase Principal**: `EconomicDataService`
**Responsabilidad**: Datos económicos de múltiples fuentes

**Métodos Principales:**
```python
async def get_market_data(symbol: str, period: str) -> Dict
async def get_economic_indicators() -> Dict
async def get_economic_outlook() -> EconomicOutlook
```

### Servicio de Correlación

**Clase Principal**: `CorrelationService`
**Responsabilidad**: Análisis estadístico avanzado

**Métodos Principales:**
```python
async def analyze_correlation(economic_data, solar_data) -> CorrelationResult
async def cross_spectral_analysis(economic_data, solar_data) -> SpectralAnalysis
async def analyze_causality(economic_data, solar_data) -> CausalAnalysis
```

## 💾 Base de Datos y Cache

### Sistema de Cache

**Clase Principal**: `CacheManager` (data/cache/__init__.py)

**Características:**
- Cache en memoria y disco
- TTL configurable por entrada
- Limpieza automática de expirados
- Claves hasheadas para seguridad

**Uso:**
```python
from data.cache import cache_manager

# Guardar en cache
cache_manager.set("clave", datos, ttl=3600)

# Recuperar de cache
datos = cache_manager.get("clave")

# Eliminar de cache
cache_manager.delete("clave")
```

### Gestión de Datos

**Datos Solares** (data/solar/):
- Formato: CSV y Parquet
- Estructura: Series temporales diarias/mensuales
- Retención: Hasta 100 años históricos

**Datos Económicos** (data/economic/):
- Formato: Parquet y JSON
- Estructura: Múltiples indicadores
- Fuentes: Yahoo Finance, FRED, Alpha Vantage

**Modelos ML** (data/models/):
- Formato: Pickle (.pkl)
- Metadatos: JSON con métricas
- Versionado: Por fecha de entrenamiento

## 🧪 Testing y Calidad

### Suite de Tests

**Estructura de Tests:**
```
tests/
├── unit/                 # Tests unitarios
│   ├── test_services.py
│   ├── test_core.py
│   └── test_api.py
├── integration/          # Tests de integración
│   ├── test_apis.py
│   └── test_services.py
└── conftest.py          # Configuración pytest
```

**Ejecutar Tests:**
```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/unit/test_services.py -v

# Con cobertura
pytest --cov=app --cov-report=html

# Tests de integración
pytest tests/integration/ -v
```

### Ejemplo de Test Unitario

```python
import pytest
from app.services.nasa_solar_service import NASASolarService

class TestNASASolarService:
    @pytest.fixture
    def solar_service(self):
        return NASASolarService()
    
    @pytest.mark.asyncio
    async def test_get_current_solar_activity(self, solar_service):
        # Configurar
        await solar_service.initialize()
        
        # Ejecutar
        result = await solar_service.get_current_solar_activity()
        
        # Verificar
        assert result is not None
        assert hasattr(result, 'sunspot_number')
        assert isinstance(result.sunspot_number, (int, float))
```

### Calidad de Código

**Herramientas:**
```bash
# Formateo de código
black app/ tests/

# Linting
flake8 app/ tests/

# Análisis de tipos
mypy app/

# Seguridad
bandit -r app/
```

**Git Hooks** (opcional):
```bash
# .git/hooks/pre-commit
#!/bin/bash
black app/ tests/
flake8 app/ tests/
pytest tests/unit/
```

## 🚀 Despliegue

### Entorno de Producción

**Requisitos:**
- Python 3.8+
- 4GB RAM mínimo
- 20GB almacenamiento
- Linux (recomendado)

**Configuración:**
```bash
# Instalar dependencias de producción
pip install -r requirements.txt

# Configurar variables de entorno de producción
export NASA_API_KEY=tu_clave_real
export FRED_API_KEY=tu_clave_real
export DEBUG=False
export LOG_LEVEL=WARNING

# Iniciar con Gunicorn (recomendado para producción)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

### Docker (Opcional)

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  heliobio-economic:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NASA_API_KEY=${NASA_API_KEY}
      - FRED_API_KEY=${FRED_API_KEY}
    volumes:
      - ./data:/app/data
```

## 🤝 Contribución

### Proceso de Contribución

1. **Fork** del repositorio
2. **Feature Branch**: `git checkout -b feature/nueva-funcionalidad`
3. **Desarrollo**: Seguir convenciones de código
4. **Tests**: Añadir tests para nueva funcionalidad
5. **Documentación**: Actualizar manuales relevantes
6. **Pull Request**: Descripción detallada de cambios

### Áreas de Contribución Prioritaria

**Alta Prioridad:**
- Mejora de modelos ML (LSTM, Transformers)
- Integración de nuevas fuentes de datos
- Optimización de performance
- Mejora de documentación

**Media Prioridad:**
- Nuevos métodos de análisis
- Visualizaciones avanzadas
- Sistema de alertas
- APIs adicionales

**Baja Prioridad:**
- Refactorizaciones menores
- Mejoras de UI/UX
- Traducciones

### Estándares de Commits

**Formato**: Conventional Commits
```
feat: añadir análisis wavelet para correlaciones
fix: corregir error en cálculo de correlación de distancia
docs: actualizar manual de usuario con nuevos endpoints
test: añadir tests para servicio NASA
refactor: mejorar estructura de servicios de datos
```

### Revisión de Código

**Checklist para PR:**
- [ ] Código sigue convenciones establecidas
- [ ] Tests pasan y cobertura adecuada
- [ ] Documentación actualizada
- [ ] Manuales de usuario actualizados
- [ ] No introduce breaking changes
- [ ] Logs apropiados añadidos

## 🐛 Debugging y Troubleshooting

### Debugging en Desarrollo

**Logs Detallados:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Debugger Integrado:**
```python
import pdb; pdb.set_trace()  # Punto de interrupción
```

**Herramientas:**
```bash
# Profile de performance
python -m cProfile -o profile.stats app/main.py

# Análisis de memoria
pip install memory_profiler
python -m memory_profiler app/main.py
```

### Monitoreo en Producción

**Métricas Clave:**
- Tiempo de respuesta API
- Uso de memoria y CPU
- Tasa de errores
- Estado de servicios externos

**Health Checks:**
```bash
curl http://localhost:8000/api/system/health
```

---

**¿Preguntas?** 
- 📧 Email: ia.mechmind@gmail.com
- 🐛 Issues: GitHub Issues
- 💬 Discusiones: GitHub Discussions

*¡Gracias por contribuir a HelioBio-Economic! 🌟*
