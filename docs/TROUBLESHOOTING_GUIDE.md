# 🛠️ Guía de Solución de Problemas - HelioBio-Economic
**Para la versión 1.0.0** | *Última actualización: {{fecha_actual}}*
Guía completa para diagnosticar y resolver problemas comunes en HelioBio-Economic.
## 🚨 Síntomas Rápidos y Soluciones
### El servidor no inicia
**Síntoma:**
```bash
Error: ModuleNotFoundError: No module named 'fastapi'
```
**Solución:**
```bash
# 1. Verificar que estás en el entorno virtual
source helio_env/bin/activate  # Linux/Mac
# helio_env\Scripts\activate   # Windows
# 2. Instalar dependencias
pip install -r requirements.txt
# 3. Verificar instalación
python -c "import fastapi; print('FastAPI OK')"
```
### La API responde con error 500
**Síntoma:**
```bash
curl http://localhost:8000/api/solar/current
# {"success":false,"message":"Error interno del servidor"}
```
**Solución:**
```bash
# 1. Verificar logs del servidor
tail -f logs/helio_economic.log
# 2. Verificar estado de APIs externas
curl http://localhost:8000/api/system/health
# 3. Reiniciar el servidor
pkill -f uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
### Los datos económicos no se cargan
**Síntoma:**
```bash
curl http://localhost:8000/api/economic/indicators
# {"success":false,"error":"FRED API no disponible"}
```
**Solución:**
```bash
# 1. Verificar configuración de API keys
cat .env | grep FRED
# 2. Probar conexión a FRED directamente
python -c "
from fredapi import Fred
fred = Fred(api_key='TU_CLAVE_FRED')
print(fred.get_series('GDP', limit=1))
"
# 3. Usar datos de muestra
# Editar app/services/economic_data_service.py
# Cambiar use_sample_data = True temporalmente
```
## 🔍 Diagnóstico Detallado
### 1. Verificación del Sistema
**Paso 1: Estado General**
```bash
# Verificar salud completa del sistema
curl -s http://localhost:8000/api/system/health | jq '.data'
# Respuesta esperada:
{
  "system_status": "healthy",
  "services": {
    "nasa_solar_service": {"status": "healthy"},
    "economic_data_service": {"status": "healthy"}
  }
}
```
**Paso 2: Estado Detallado**
```bash
# Información detallada de componentes
curl -s http://localhost:8000/api/system/status | jq '.data'
# Verificar específicamente:
# - NASA API disponible
# - Modelos ML entrenados  
# - APIs económicas activas
```
**Paso 3: Métricas del Sistema**
```bash
# Verificar uso de recursos
curl -s http://localhost:8000/api/system/health | jq '.data.system_metrics'
# Umbrales críticos:
# - CPU > 90%: Posible cuello de botella
# - Memoria > 85%: Riesgo de crash
# - Disco > 95%: Espacio insuficiente
```
### 2. Problemas de APIs Externas
#### NASA DONKI API
**Síntomas:**
- Datos solares vacíos o desactualizados
- Error "NASA API no disponible"
**Diagnóstico:**
```bash
# Probar conexión directa a NASA API
curl "https://api.nasa.gov/DONKI/FLR?apiKey=DEMO_KEY..."
# Verificar en logs
grep "NASA" logs/helio_economic.log | tail -10
```
**Soluciones:**
```bash
# 1. Verificar API key
echo "NASA_API_KEY=$NASA_API_KEY"
# 2. Usar clave demo temporal
export NASA_API_KEY=DEMO_KEY
# 3. Verificar límites de rate limiting
# NASA limita a 1000 requests por hora
```
#### Yahoo Finance API
**Síntomas:**
- Datos de mercado vacíos
- Símbolos no encontrados
**Diagnóstico:**
```python
# Probar yfinance directamente
import yfinance as yf
ticker = yf.Ticker("^GSPC")
print(ticker.history(period="1mo"))
```
**Soluciones:**
```bash
# 1. Verificar conexión a internet
ping api.finance.yahoo.com
# 2. Actualizar yfinance
pip install --upgrade yfinance
# 3. Usar símbolos alternativos
# ^GSPC (S&P 500), ^DJI (Dow Jones), etc.
```
#### FRED API
**Síntomas:**
- Indicadores económicos vacíos
- Error "FRED API key not configured"
**Diagnóstico:**
```bash
# Verificar configuración
grep FRED_API_KEY .env
# Probar conexión
python -c "
from fredapi import Fred
fred = Fred(api_key='$FRED_API_KEY') 
print('Series disponibles:', fred.get_series('GDP').head(2))
"
```
**Soluciones:**
```bash
# 1. Obtener API key de FRED
# Visitar: https://research.stlouisfed.org/docs/api/api_key.html
# 2. Configurar en .env
echo "FRED_API_KEY=tu_clave_aqui" >> .env
# 3. Reiniciar servidor
```
### 3. Problemas de Modelos ML
#### Modelos No Entrenados
**Síntoma:**
```bash
curl http://localhost:8000/api/prediction/economic
# {"error": "Modelos no entrenados"}
```
**Solución:**
```bash
# Entrenar modelos manualmente
curl -X POST http://localhost:8000/api/prediction/retrain-models
# Verificar progreso en logs
tail -f logs/helio_economic.log | grep "ML"
```
#### Bajo Rendimiento de Modelos
**Síntoma:**
```bash
curl http://localhost:8000/api/prediction/ml-performance
# R² scores bajos (< 0.3)
```
**Diagnóstico:**
```bash
# Verificar métricas de todos los modelos
curl -s http://localhost:8000/api/prediction/ml-performance | jq '.data.model_performance'
```
**Soluciones:**
```python
# 1. Aumentar datos de entrenamiento
# Editar: app/core/solar_economic_ml.py
# Cambiar: years=100 en get_historical_solar_data()
# 2. Ajustar hiperparámetros
# En model_configs, aumentar n_estimators, max_depth, etc.
# 3. Agregar más características
# En _create_engineered_features(), añadir más lags e interacciones
```
#### Sobreentrenamiento (Overfitting)
**Síntomas:**
- R² alto en entrenamiento, bajo en validación
- Predicciones poco realistas
**Solución:**
```python
# 1. Aumentar regularización
'model_configs': {
    'random_forest_advanced': {
        'params': {
            'max_depth': 10,  # Reducir de 15
            'min_samples_split': 10,  # Aumentar de 5
            'min_samples_leaf': 4  # Aumentar de 2
        }
    }
}
# 2. Añadir validación cruzada más estricta
'cross_validation_folds': 10  # Aumentar de 5
```
### 4. Problemas de Rendimiento
#### Lento para Cargar Datos
**Síntoma:**
- Requests toman más de 10 segundos
- Timeouts frecuentes
**Diagnóstico:**
```bash
# Medir tiempo de respuesta
time curl -s http://localhost:8000/api/solar/historical?years=50 > /dev/null
# Verificar caché
curl -s http://localhost:8000/api/system/status | jq '.data.cache_stats'
```
**Soluciones:**
```python
# 1. Optimizar caché (en cada servicio)
self.cache_duration = timedelta(hours=1)  # Aumentar duración
# 2. Reducir datos históricos por defecto
years: int = Query(30, description="Años de datos")  # Reducir de 50
# 3. Implementar paginación
# Para endpoints con muchos datos
```
#### Alto Uso de Memoria
**Síntoma:**
- Servidor se vuelve lento con el tiempo
- Crash por "out of memory"
**Diagnóstico:**
```bash
# Monitorear uso de memoria
ps aux | grep uvicorn | awk '{print $5}'
# Verificar leaks de memoria
python -m memory_profiler app/main.py
```
**Soluciones:**
```python
# 1. Limitar tamaño de caché
MAX_CACHE_SIZE = 1000  # Máximo de items en caché
# 2. Limpiar caché periódicamente
def cleanup_old_cache():
    now = datetime.now()
    for key in list(self.cache.keys()):
        if now - self.cache[key]['timestamp'] > self.cache_duration:
            del self.cache[key]
# 3. Usar datos más livianos
# Para desarrollo, usar sample_data=True
```
### 5. Problemas de Datos
#### Datos Faltantes o Incompletos
**Síntoma:**
- Series temporales con gaps
- Fechas no alineadas entre solar y económico
**Diagnóstico:**
```python
# Verificar integridad de datos
economic_data = await economic_data_service.get_long_term_economic_data()
solar_data = await nasa_solar_service.get_historical_solar_data(50)
print("Datos económicos:", economic_data.isnull().sum())
print("Datos solares:", solar_data.isnull().sum())
print("Fechas comunes:", economic_data.index.intersection(solar_data.index).size)
```
**Soluciones:**
```python
# 1. Mejorar imputación de datos
def _prepare_correlation_data(self, economic_data, solar_data):
    # Interpolación más robusta
    economic_clean = economic_data.interpolate(method='time').ffill().bfill()
    solar_clean = solar_data.interpolate(method='time').ffill().bfill()
    return economic_clean, solar_clean
# 2. Sincronizar frecuencias
# Convertir ambas series a frecuencia mensual
economic_monthly = economic_data.resample('M').mean()
solar_monthly = solar_data.resample('M').mean()
```
#### Datos Desactualizados
**Síntoma:**
- Datos solares con varias horas de retraso
- Indicadores económicos del mes anterior
**Solución:**
```python
# 1. Reducir intervalo de actualización
self.cache_duration = timedelta(minutes=5)  # Para datos en tiempo real
# 2. Implementar actualización en background
async def background_data_refresh():
    while True:
        await asyncio.sleep(300)  # 5 minutos
        await self._refresh_current_data()
```
## 🛠️ Procedimientos de Mantenimiento
### Backup de Datos y Modelos
```bash
#!/bin/bash
# scripts/backup_system.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/backup_$DATE"
mkdir -p $BACKUP_DIR
# Backup de datos
cp -r data/ $BACKUP_DIR/
# Backup de modelos entrenados
cp -r models/ $BACKUP_DIR/ 2>/dev/null || echo "No models directory"
# Backup de configuración
cp .env $BACKUP_DIR/
cp logs/helio_economic.log $BACKUP_DIR/
echo "Backup completado: $BACKUP_DIR"
```
### Limpieza de Caché y Logs
```bash
#!/bin/bash
# scripts/cleanup_system.sh
# Limpiar caché viejo
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
# Rotar logs (mantener últimos 7 días)
find logs/ -name "*.log" -mtime +7 -delete
# Limpiar caché de datos temporales
rm -rf data/temp/* 2>/dev/null
echo "Limpieza del sistema completada"
```
### Actualización del Sistema
```bash
#!/bin/bash
# scripts/update_system.sh
echo "=== Actualizando HelioBio-Economic ==="
# 1. Backup actual
./scripts/backup_system.sh
# 2. Actualizar código
git pull origin main
# 3. Actualizar dependencias
pip install -r requirements.txt --upgrade
# 4. Re-entrenar modelos
curl -X POST http://localhost:8000/api/prediction/retrain-models
# 5. Reiniciar servidor
pkill -f uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
echo "Sistema actualizado correctamente"
```
## 📊 Monitoreo y Alertas
### Script de Monitoreo Automático
```python
# scripts/monitor_system.py
import requests
import time
import logging
from datetime import datetime
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
def check_system_health():
    try:
        response = requests.get('http://localhost:8000/api/system/health', timeout=10)
        data = response.json()
        if data['success']:
            status = data['data']['system_status']
            metrics = data['data']['system_metrics']
            # Alertas por umbrales
            if metrics['cpu_percent'] > 80:
                logging.warning(f"CPU alta: {metrics['cpu_percent']}%")
            if metrics['memory_usage'] > 85:
                logging.warning(f"Memoria alta: {metrics['memory_usage']}%")
            if status != 'healthy':
                logging.error(f"Sistema no saludable: {status}")
            return True
        else:
            logging.error("Health check falló")
            return False
    except Exception as e:
        logging.error(f"Error en monitoreo: {e}")
        return False
if __name__ == "__main__":
    while True:
        check_system_health()
        time.sleep(300)  # Verificar cada 5 minutos
```
### Configuración de Alertas
```bash
# Configurar alertas por email (ejemplo)
# En scripts/alert_system.py
import smtplib
from email.mime.text import MimeText
def send_alert(subject, message):
    msg = MimeText(message)
    msg['Subject'] = f"[HelioBio-Economic] {subject}"
    msg['From'] = 'alerts@heliobio.com'
    msg['To'] = 'admin@heliobio.com'
    # Configurar servidor SMTP
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
```
## 🔄 Recuperación ante Fallos
### Procedimiento de Restauración
**Caso: Servidor crasheado**
```bash
# 1. Verificar causa del crash
tail -n 100 logs/helio_economic.log
# 2. Restaurar desde backup más reciente
LATEST_BACKUP=$(ls -td backups/* | head -1)
cp -r $LATEST_BACKUP/data/ ./
cp $LATEST_BACKUP/.env ./
# 3. Reiniciar servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
**Caso: Datos corruptos**
```bash
# 1. Detener servidor
pkill -f uvicorn
# 2. Limpiar caché corrupto
rm -rf data/cache/*
# 3. Re-descargar datos
curl -X POST http://localhost:8000/api/prediction/retrain-models
# 4. Reiniciar
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
## 📞 Soporte Técnico
### Información para Reportar Problemas
Al contactar soporte, incluir:
1. **Versión del sistema:**
   ```bash
   curl -s http://localhost:8000/api/system/health | jq '.data.version'
   ```
2. **Logs relevantes:**
   ```bash
   tail -n 50 logs/helio_economic.log
   ```
3. **Estado del sistema:**
   ```bash
   curl -s http://localhost:8000/api/system/status | jq '.data'
   ```
4. **Configuración:**
   ```bash
   cat .env | grep -v "KEY\|PASSWORD"
   ```
### Canales de Soporte
- **📧 Email**: ia.mechmind@gmail.com
- **🐛 GitHub Issues**: [github.com/mechmind-dwv/HelioBio-Economic/issues](https://github.com/mechmind-dwv/HelioBio-Economic/issues)
- **📚 Documentación**: [localhost:8000/docs](http://localhost:8000/docs)
## 🎯 Checklist de Verificación Rápida
Antes de contactar soporte, verificar:
- [ ] Servidor ejecutándose en puerto 8000
- [ ] Entorno virtual activado
- [ ] Todas las dependencias instaladas
- [ ] APIs externas accesibles
- [ ] Espacio suficiente en disco
- [ ] Última versión del código
---
**¿Problema no resuelto?** Contacta al equipo de soporte con la información de diagnóstico completa. 
*HelioBio-Economic - Conectando cosmos y economía* 🌞💹
