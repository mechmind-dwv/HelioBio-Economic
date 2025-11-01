# 🌞 HelioBio-Economic API Reference
**Versión 1.0.0** | *Última actualización: {{fecha_actual}}*
Documentación completa de la API REST de HelioBio-Economic para análisis de correlación solar-económica.
## 🔗 Base URL
```
http://localhost:8000/api
```
## 📊 Resumen de Endpoints
### 🌞 Datos Solares
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/solar/current` | Actividad solar actual |
| `GET` | `/solar/historical` | Datos solares históricos |
| `GET` | `/solar/flares` | Fulguraciones solares recientes |
| `GET` | `/solar/cme` | Eyecciones de Masa Coronal |
| `GET` | `/solar/forecast` | Pronóstico solar |
### 💹 Datos Económicos
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/economic/markets` | Datos de mercados bursátiles |
| `GET` | `/economic/indicators` | Indicadores macroeconómicos |
| `GET` | `/economic/kondratiev` | Análisis ondas largas |
| `GET` | `/economic/conditions` | Condiciones de mercado |
| `GET` | `/economic/outlook` | Perspectiva económica |
### 🔗 Análisis de Correlación
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/correlation/solar-economic` | Correlación solar-económica |
| `GET` | `/correlation/cross-spectral` | Análisis espectral cruzado |
| `GET` | `/correlation/causality` | Análisis de causalidad |
| `GET` | `/correlation/common-cycles` | Ciclos comunes |
### 🎯 Predicciones
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/prediction/economic` | Predicción ciclos económicos |
| `GET` | `/prediction/ml-performance` | Métricas modelos ML |
| `POST` | `/prediction/retrain-models` | Re-entrenar modelos |
| `GET` | `/prediction/crisis-risk` | Riesgo de crisis |
### ⚙️ Sistema
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/system/health` | Estado del sistema |
| `GET` | `/system/status` | Estado detallado |
## 🌞 Endpoints de Datos Solares
### GET /solar/current
Obtener actividad solar actual consolidada de NASA DONKI, NOAA SWPC y SILSO.
**Parámetros:** Ninguno
**Respuesta:**
```json
{
  "success": true,
  "message": "Actividad solar actual obtenida correctamente",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "solar_activity": {
      "timestamp": "2024-01-15T10:30:00Z",
      "sunspot_number": 67,
      "solar_flux": 145.2,
      "kp_index": 3.2,
      "wind_speed": 450.5,
      "proton_flux": 2.1,
      "electron_flux": 345.6,
      "xray_flux": 1.2e-5,
      "geomagnetic_field": "quiet"
    },
    "data_sources": ["NASA DONKI", "NOAA SWPC", "SILSO"],
    "update_frequency": "60 seconds"
  }
}
```
### GET /solar/historical
Obtener datos solares históricos.
**Parámetros Query:**
- `years` (int, opcional): Años de datos históricos (1-100). Default: 50
- `include_cycles` (bool, opcional): Incluir ciclos solares identificados. Default: true
**Ejemplo:**
```bash
curl "http://localhost:8000/api/solar/historical?years=30&include_cycles=true"
```
**Respuesta:**
```json
{
  "success": true,
  "message": "Datos solares históricos de 30 años obtenidos correctamente",
  "data": {
    "period_years": 30,
    "data_points": 360,
    "date_range": {
      "start": "1994-01-01T00:00:00Z",
      "end": "2024-01-01T00:00:00Z"
    },
    "historical_data": [...],
    "solar_cycles": [
      {
        "cycle_number": 23,
        "start_date": "1996-08-01T00:00:00Z",
        "end_date": "2008-12-01T00:00:00Z",
        "peak_date": "2000-03-01T00:00:00Z",
        "period_years": 11.3,
        "amplitude": 120.8,
        "confidence": 0.95
      }
    ]
  }
}
```
### GET /solar/flares
Obtener fulguraciones solares recientes.
**Parámetros Query:**
- `days` (int, opcional): Días hacia atrás (1-30). Default: 7
**Ejemplo:**
```bash
curl "http://localhost:8000/api/solar/flares?days=14"
```
## 💹 Endpoints de Datos Económicos
### GET /economic/markets
Obtener datos de mercados bursátiles.
**Parámetros Query:**
- `symbol` (string, opcional): Símbolo del mercado. Default: "^GSPC"
- `period` (string, opcional): Período de datos. Default: "1y"
**Símbolos disponibles:**
- `^GSPC` - S&P 500
- `^DJI` - Dow Jones Industrial Average
- `^IXIC` - NASDAQ Composite
- `^RUT` - Russell 2000
- `^FTSE` - FTSE 100
- `^GDAXI` - DAX
- `^N225` - Nikkei 225
**Períodos disponibles:**
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`
**Ejemplo:**
```bash
curl "http://localhost:8000/api/economic/markets?symbol=^GSPC&period=2y"
```
**Respuesta:**
```json
{
  "success": true,
  "message": "Datos de mercado para ^GSPC obtenidos correctamente",
  "data": {
    "symbol": "^GSPC",
    "period": "2y",
    "data_points": 504,
    "current_price": 4780.35,
    "market_data": [
      {
        "symbol": "^GSPC",
        "timestamp": "2022-01-15T00:00:00Z",
        "price": 4662.85,
        "change": -15.02,
        "change_percent": -0.32,
        "volume": 3820000000,
        "open_price": 4677.87,
        "high": 4689.23,
        "low": 4650.45
      }
    ],
    "metadata": {
      "company_name": "S&P 500 Index",
      "sector": "Index",
      "currency": "USD"
    }
  }
}
```
### GET /economic/indicators
Obtener indicadores macroeconómicos principales.
**Parámetros:** Ninguno
**Respuesta:**
```json
{
  "success": true,
  "message": "Indicadores económicos obtenidos correctamente",
  "data": {
    "timestamp": "2024-01-15T10:30:00Z",
    "indicators": {
      "GDP": {
        "indicator": "GDP",
        "timestamp": "2023-10-01T00:00:00Z",
        "value": 27458.2,
        "unit": "Billions of Dollars",
        "frequency": "Quarterly",
        "previous_value": 27148.1,
        "change": 310.1,
        "change_percent": 1.14
      },
      "INFLATION": {
        "indicator": "INFLATION",
        "timestamp": "2023-12-01T00:00:00Z",
        "value": 307.051,
        "unit": "Index",
        "frequency": "Monthly",
        "previous_value": 306.746,
        "change": 0.305,
        "change_percent": 0.10
      }
    },
    "economic_health": {
      "growth": "moderate",
      "inflation": "moderate",
      "employment": "strong",
      "sentiment": "neutral"
    },
    "summary": {
      "key_highlights": ["Bajo desempleo: 3.7%", "Baja inflación: 0.10%"],
      "areas_of_concern": [],
      "outlook": "stable",
      "momentum": "positive"
    }
  }
}
```
### GET /economic/kondratiev
Análisis de ondas largas de Kondratiev.
**Parámetros:** Ninguno
**Respuesta:**
```json
{
  "success": true,
  "message": "Análisis Kondratiev completado",
  "data": {
    "current_analysis": {
      "current_wave": 6,
      "current_phase": "Primavera",
      "phase_progress": 0.35,
      "next_phase_transition": "2035-01-01T00:00:00Z"
    },
    "solar_synchronization": {
      "phase_alignment": 0.8,
      "cycle_synchronization": 0.72,
      "historical_correlation": 0.78,
      "predicted_sync_strength": 0.85
    },
    "economic_implications": {
      "growth_outlook": "Crecimiento acelerado e innovación disruptiva",
      "investment_opportunities": ["Tecnologías emergentes", "Infraestructura nueva"],
      "sector_recommendations": ["Tecnología", "Energías renovables", "Biotecnología"],
      "risk_factors": ["Sobrevaluación de innovaciones", "Regulación desfasada"]
    },
    "risk_assessment": {
      "economic_risk_level": "Bajo",
      "financial_risk_level": "Moderado",
      "social_risk_level": "Bajo",
      "technological_risk_level": "Alto",
      "composite_risk_index": 2.1
    }
  }
}
```
## 🔗 Endpoints de Análisis de Correlación
### GET /correlation/solar-economic
Análisis de correlación entre indicadores solares y económicos.
**Parámetros Query:**
- `economic_indicator` (string, opcional): Indicador económico. Default: "SP500"
- `solar_indicator` (string, opcional): Indicador solar. Default: "sunspots"
- `period_years` (int, opcional): Período de análisis (10-100). Default: 50
**Indicadores económicos disponibles:**
- `SP500` - S&P 500 Index
- `GDP` - Producto Interno Bruto
- `INFLATION` - Índice de Precios al Consumidor
- `UNEMPLOYMENT` - Tasa de Desempleo
- `INTEREST_RATE` - Tasa de Fondos Federales
**Indicadores solares disponibles:**
- `sunspots` - Número de manchas solares
- `solar_flux` - Flujo solar
- `kp_index` - Índice Kp geomagnético
- `wind_speed` - Velocidad del viento solar
**Ejemplo:**
```bash
curl "http://localhost:8000/api/correlation/solar-economic?economic_indicator=SP500&solar_indicator=sunspots&period_years=30"
```
**Respuesta:**
```json
{
  "success": true,
  "message": "Correlación SP500-sunspots analizada",
  "data": {
    "analysis_parameters": {
      "economic_indicator": "SP500",
      "solar_indicator": "sunspots",
      "period_years": 30,
      "sample_size": 360
    },
    "correlation_results": {
      "methods": {
        "pearson": 0.324,
        "spearman": 0.298,
        "kendall": 0.215,
        "mutual_information": 0.187,
        "distance_correlation": 0.276
      },
      "optimal_lag_months": -6,
      "lag_correlation": 0.356,
      "p_value": 0.012,
      "confidence_interval": [0.245, 0.398],
      "significance": "Moderada"
    },
    "statistical_tests": {
      "stationarity": {
        "economic_adf": {"is_stationary": true, "p_value": 0.001},
        "solar_adf": {"is_stationary": true, "p_value": 0.003},
        "both_stationary": true
      }
    },
    "interpretation": {
      "notes": [
        "Correlación Moderada",
        "Lag significativo: -6 meses",
        "Relación monotónica más fuerte que lineal"
      ],
      "strength": "Moderada",
      "direction": "Positive"
    }
  }
}
```
### GET /correlation/cross-spectral
Análisis espectral cruzado entre ciclos solares y económicos.
**Parámetros:** Ninguno
**Respuesta:**
```json
{
  "success": true,
  "message": "Análisis espectral cruzado completado",
  "data": {
    "spectral_analysis": {
      "common_periods": [11.2, 22.1, 54.3, 87.5],
      "coherence_spectrum": {
        "11.2": 0.78,
        "22.1": 0.65,
        "54.3": 0.72,
        "87.5": 0.58
      },
      "phase_synchronization": {
        "11.2": 0.82,
        "22.1": 0.71,
        "54.3": 0.75,
        "87.5": 0.63
      },
      "shared_cycles": [
        {
          "period_years": 11.2,
          "coherence_strength": 0.78,
          "cycle_type": "Solar Schwabe",
          "significance": "Alta",
          "theoretical_match": "Solar 11 Year"
        }
      ],
      "dominant_frequencies": [
        {
          "series": "economic",
          "period_years": 9.8,
          "power": 0.045,
          "cycle_type": "Juglar"
        }
      ]
    }
  }
}
```
## 🎯 Endpoints de Predicción
### GET /prediction/economic
Predecir ciclos económicos basados en factores solares.
**Parámetros Query:**
- `cycles` (string, opcional): Ciclos a incluir. Default: "solar,kondratiev"
- `horizon_days` (int, opcional): Horizonte de predicción (1-365). Default: 30
**Ciclos disponibles:**
- `solar` - Ciclos solares
- `kondratiev` - Ondas largas
- `kuznets` - Ciclos de construcción
- `juglar` - Ciclos de inversión
- `kitchin` - Ciclos de inventarios
**Ejemplo:**
```bash
curl "http://localhost:8000/api/prediction/economic?cycles=solar,kondratiev,kuznets&horizon_days=90"
```
**Respuesta:**
```json
{
  "success": true,
  "message": "Predicción económica para 90 días generada",
  "data": {
    "individual_predictions": {
      "random_forest_advanced": {
        "predictions": [4780.35, 4812.67, 4856.23, ...],
        "confidence_interval": [4756.12, 4890.45],
        "model_performance": {"r2_score": 0.412, "rmse": 45.67},
        "solar_influence": 0.324,
        "trend": "bullish"
      }
    },
    "ensemble_prediction": {
      "predictions": [4785.23, 4820.45, 4867.89, ...],
      "ensemble_method": "weighted_average_by_r2",
      "models_combined": 5
    },
    "horizon_days": 90,
    "prediction_timestamp": "2024-01-15T10:30:00Z",
    "model_consensus": {
      "bullish_consensus": 0.6,
      "bearish_consensus": 0.2,
      "neutral_consensus": 0.2,
      "dominant_trend": "bullish",
      "agreement_level": 0.6
    }
  }
}
```
### POST /prediction/retrain-models
Re-entrenar modelos de Machine Learning.
**Parámetros:** Ninguno
**Respuesta:**
```json
{
  "success": true,
  "message": "Modelos ML re-entrenados correctamente",
  "data": {
    "training_performance": {
      "random_forest_advanced": {
        "r2_score": 0.412,
        "rmse": 45.67,
        "mae": 32.45,
        "training_time_seconds": 12.3
      }
    },
    "models_trained": 5,
    "best_model": "random_forest_advanced"
  }
}
```
## ⚙️ Endpoints del Sistema
### GET /system/health
Verificar estado de salud del sistema completo.
**Parámetros:** Ninguno
**Respuesta:**
```json
{
  "success": true,
  "message": "Estado del sistema verificado",
  "data": {
    "system_status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "services": {
      "nasa_solar_service": {
        "status": "healthy",
        "nasa_api_available": true,
        "api_key_configured": true
      },
      "economic_data_service": {
        "status": "healthy",
        "yahoo_finance": {"status": "healthy"},
        "fred_api": {"status": "healthy"}
      }
    },
    "system_metrics": {
      "cpu_percent": 45.2,
      "memory_usage": 67.8,
      "disk_usage": 23.4,
      "process_memory_mb": 245.6
    },
    "version": "HelioBio-Economic v1.0.0"
  }
}
```
## 🔐 Autenticación
Actualmente la API no requiere autenticación para desarrollo. Para entornos de producción, se recomienda implementar:
```python
# Ejemplo de implementación futura
API_KEYS = {
    "your-api-key": "user@example.com"
}
```
## 🚦 Códigos de Estado HTTP
| Código | Descripción |
|--------|-------------|
| `200` | OK - Solicitud exitosa |
| `400` | Bad Request - Parámetros inválidos |
| `404` | Not Found - Recurso no encontrado |
| `500` | Internal Server Error - Error del servidor |
| `503` | Service Unavailable - Servicio temporalmente no disponible |
## 📈 Ejemplos de Uso
### Ejemplo 1: Análisis de Correlación Completo
```bash
#!/bin/bash
BASE_URL="http://localhost:8000/api"
# 1. Obtener datos actuales
echo "=== DATOS ACTUALES ==="
curl -s "$BASE_URL/solar/current" | jq '.data.solar_activity'
curl -s "$BASE_URL/economic/indicators" | jq '.data.indicators.GDP'
# 2. Análisis de correlación
echo "=== ANÁLISIS DE CORRELACIÓN ==="
curl -s "$BASE_URL/correlation/solar-economic?economic_indicator=SP500&solar_indicator=sunspots&period_years=30" | jq '.data.correlation_results'
# 3. Predicción
echo "=== PREDICCIÓN ==="
curl -s "$BASE_URL/prediction/economic?cycles=solar,kondratiev&horizon_days=30" | jq '.data.ensemble_prediction'
```
### Ejemplo 2: Monitoreo Continuo
```python
import requests
import time
import json
def monitor_heliobio_system():
    base_url = "http://localhost:8000/api"
    while True:
        try:
            # Verificar salud del sistema
            health = requests.get(f"{base_url}/system/health").json()
            if health['data']['system_status'] == 'healthy':
                # Obtener correlación actual
                correlation = requests.get(
                    f"{base_url}/correlation/solar-economic",
                    params={'period_years': 10}
                ).json()
                print(f"Correlación actual: {correlation['data']['correlation_results']['methods']['pearson']:.3f}")
            time.sleep(300)  # Esperar 5 minutos
        except Exception as e:
            print(f"Error en monitoreo: {e}")
            time.sleep(60)
```
## 🐛 Solución de Problemas
### Error Comunes
1. **API no responde**
   ```bash
   # Verificar que el servidor esté ejecutándose
   curl http://localhost:8000/api/system/health
   ```
2. **Datos faltantes**
   ```bash
   # Verificar conexión a APIs externas
   curl http://localhost:8000/api/system/status | jq '.data.services'
   ```
3. **Modelos no entrenados**
   ```bash
   # Re-entrenar modelos
   curl -X POST http://localhost:8000/api/prediction/retrain-models
   ```
## 📞 Soporte
Para reportar problemas o solicitar características:
1. **Issues GitHub**: [github.com/mechmind-dwv/HelioBio-Economic/issues](https://github.com/mechmind-dwv/HelioBio-Economic/issues)
2. **Email**: ia.mechmind@gmail.com
3. **Documentación**: [localhost:8000/docs](http://localhost:8000/docs)
