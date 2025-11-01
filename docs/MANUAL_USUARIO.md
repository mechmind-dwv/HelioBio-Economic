# 🌞 Manual de Usuario - HelioBio-Economic v1.0

## 📖 Índice
1. [Introducción](#introducción)
2. [Instalación Rápida](#instalación-rápida)
3. [Primeros Pasos](#primeros-pasos)
4. [Dashboard Principal](#dashboard-principal)
5. [Análisis de Datos](#análisis-de-datos)
6. [Interpretación de Resultados](#interpretación-de-resultados)
7. [Casos de Uso](#casos-de-uso)
8. [Solución de Problemas](#solución-de-problemas)

## 🎯 Introducción

**HelioBio-Economic** es un sistema innovador que analiza las correlaciones entre la actividad solar y los ciclos económicos. Basado en las investigaciones de Alexander Chizhevsky y Nikolai Kondratiev, este sistema te permite:

- 📊 **Monitorear** actividad solar en tiempo real
- 💹 **Analizar** indicadores económicos globales
- 🔗 **Descubrir** correlaciones ocultas solares-económicas
- 🎯 **Predecir** tendencias basadas en ciclos históricos

### Público Objetivo
- **Investigadores** en economía y astrofísica
- **Analistas financieros** y gestores de riesgo
- **Estudiantes** de ciencias económicas y físicas
- **Entusiastas** de los ciclos económicos y solares

## 🚀 Instalación Rápida

### Requisitos del Sistema
- **Sistema Operativo**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 o superior
- **Memoria RAM**: 8GB mínimo (16GB recomendado)
- **Almacenamiento**: 5GB de espacio libre
- **Conexión Internet**: Para datos en tiempo real

### Instalación en 4 Pasos

**Paso 1: Descargar el Proyecto**
```bash
git clone https://github.com/mechmind-dwv/HelioBio-Economic.git
cd HelioBio-Economic
```

**Paso 2: Instalación Automática**
```bash
# Ejecutar script de instalación (Linux/macOS)
chmod +x scripts/install.sh
./scripts/install.sh

# Windows - Ejecutar en PowerShell
scripts/install.bat
```

**Paso 3: Configurar APIs**
```bash
# Configuración interactiva
python scripts/setup_apis.py
```

**Paso 4: Iniciar la Aplicación**
```bash
# Activar entorno virtual
source helio_env/bin/activate  # Linux/macOS
helio_env\Scripts\activate     # Windows

# Iniciar servidor
python app/main.py
```

**Paso 5: Acceder al Sistema**
Abre tu navegador y visita: `http://localhost:8000`

## 🎮 Primeros Pasos

### Pantalla de Bienvenida
Al acceder al sistema, verás el **Dashboard Principal** con:

- **Header**: Estado del sistema y título
- **Pestañas de Navegación**: 5 secciones principales
- **Tarjetas de Métricas**: Datos clave en tiempo real
- **Gráficos Interactivos**: Visualizaciones dinámicas

### Navegación Principal
1. **📊 Dashboard** - Vista general del sistema
2. **🌞 Datos Solares** - Actividad solar en tiempo real
3. **💹 Datos Económicos** - Indicadores financieros
4. **🔗 Correlaciones** - Análisis de relaciones
5. **🎯 Predicciones** - Modelos predictivos

## 📊 Dashboard Principal

### Tarjeta: Actividad Solar Actual
**Qué muestra:**
- Número de manchas solares
- Flujo solar (SFU)
- Índice Kp (actividad geomagnética)
- Velocidad del viento solar

**Cómo interpretar:**
- **>50 manchas**: Alta actividad solar
- **Kp > 5**: Tormenta geomagnética
- **Flujo > 150**: Mayor radiación UV

### Tarjeta: Indicadores Económicos
**Qué muestra:**
- S&P 500 en tiempo real
- Tendencia del mercado
- Volatilidad reciente

**Cómo interpretar:**
- **Tendencia ↗️**: Mercado alcista
- **Tendencia ↘️**: Mercado bajista
- **Volatilidad alta**: Mayor riesgo

### Tarjeta: Correlación Solar-Económica
**Qué muestra:**
- Coeficiente de correlación Pearson
- Fuerza de la relación
- Significancia estadística

**Cómo interpretar:**
- **0.7-1.0**: Correlación fuerte
- **0.4-0.7**: Correlación moderada
- **0.0-0.4**: Correlación débil
- **Valor negativo**: Relación inversa

## 🔍 Análisis de Datos

### 1. Análisis Solar en Tiempo Real

**Acceso**: Pestaña "🌞 Datos Solares"

**Datos disponibles:**
- Fulguraciones solares (últimos 7 días)
- Eyecciones de Masa Coronal (CMEs)
- Tormentas geomagnéticas
- Datos históricos (hasta 100 años)

**Ejemplo de uso:**
```bash
# API Endpoint para fulguraciones
curl "http://localhost:8000/api/solar/flares?days=3"
```

### 2. Análisis Económico

**Acceso**: Pestaña "💹 Datos Económicos"

**Indicadores disponibles:**
- Mercados bursátiles (S&P 500, NASDAQ, etc.)
- Indicadores macroeconómicos (PIB, inflación, empleo)
- Ciclos de Kondratiev (ondas largas)

**Ejemplo de uso:**
```bash
# API Endpoint para S&P 500
curl "http://localhost:8000/api/economic/markets?symbol=^GSPC&period=1y"
```

### 3. Análisis de Correlación

**Acceso**: Pestaña "🔗 Correlaciones"

**Métodos disponibles:**
- Correlación de Pearson (lineal)
- Correlación de Spearman (monotónica)
- Información Mutua (no lineal)
- Análisis espectral (ciclos comunes)

**Ejemplo de uso:**
```bash
# Análisis de correlación
curl "http://localhost:8000/api/correlation/solar-economic?economic_indicator=SP500&solar_indicator=sunspots&period_years=20"
```

## 📈 Interpretación de Resultados

### Correlaciones Significativas

**Alta Correlación (>0.7):**
- Fuerte evidencia de relación
- Posible valor predictivo
- Recomendado para análisis profundo

**Correlación Moderada (0.4-0.7):**
- Relación interesante
- Merece investigación adicional
- Considerar otros factores

**Correlación Débil (<0.4):**
- Relación probablemente casual
- Poco valor predictivo
- Considerar ruido estadístico

### Ciclos de Kondratiev

**Fases identificadas:**
1. **Primavera** (Expansión): Innovación, crecimiento
2. **Verano** (Prosperidad): Madurez, sobreinversión
3. **Otoño** (Estancamiento): Crisis financieras
4. **Invierno** (Depresión): Reinvención, nuevos paradigmas

**Cómo usar esta información:**
- **Inversores**: Ajustar estrategias por fase
- **Investigadores**: Estudiar patrones históricos
- **Políticos**: Preparar políticas contracíclicas

## 🎯 Casos de Uso Prácticos

### Caso 1: Gestión de Riesgo de Inversión

**Objetivo**: Reducir exposición durante tormentas solares intensas

**Pasos:**
1. Monitorear alertas solares en dashboard
2. Verificar correlación con volatilidad del mercado
3. Ajustar cartera si correlación es fuerte
4. Monitorear indicadores de recuperación

**Endpoint útil:**
```bash
/api/solar/current
/api/economic/conditions
```

### Caso 2: Investigación Académica

**Objetivo**: Estudiar relación entre máximos solares y recesiones

**Pasos:**
1. Obtener datos históricos (50+ años)
2. Realizar análisis de correlación cruzada
3. Identificar lags temporales significativos
4. Publicar hallazgos con significancia estadística

**Endpoint útil:**
```bash
/api/correlation/solar-economic
/api/economic/kondratiev
```

### Caso 3: Alerta Temprana para Empresas

**Objetivo**: Anticipar cambios en sentimiento del consumidor

**Pasos:**
1. Establecer línea base de correlación
2. Configurar alertas para cambios significativos
3. Integrar con sistemas internos de analytics
4. Ajustar estrategias de marketing

## 🛠️ Solución de Problemas

### Problemas Comunes

**1. Error de Conexión API**
```
Síntoma: "Error de conexión" en dashboard
Solución:
- Verificar conexión a internet
- Revisar claves API en .env
- Ejecutar: /api/system/health
```

**2. Datos No Actualizados**
```
Síntoma: Métricas muestran "--" o datos antiguos
Solución:
- Verificar logs en terminal
- Revisar estado de servicios externos
- Reiniciar aplicación
```

**3. Gráficos No se Cargan**
```
Síntoma: Espacios en blanco en lugar de gráficos
Solución:
- Verificar JavaScript en navegador
- Actualizar navegador a versión reciente
- Probar en modo incógnito
```

**4. Alto Uso de CPU/Memoria**
```
Síntoma: Aplicación lenta o que se cuelga
Solución:
- Reducir frecuencia de actualización
- Limitar años de datos históricos
- Aumentar recursos del sistema
```

### Comandos de Diagnóstico

```bash
# Verificar estado del sistema
curl http://localhost:8000/api/system/health

# Ver logs de aplicación
tail -f logs/helio_economic.log

# Probar conexión NASA
curl "https://api.nasa.gov/DONKI/FLR?apiKey=DEMO_KEY"

# Probar conexión Yahoo Finance
python -c "import yfinance as yf; print(yf.Ticker('^GSPC').info['currentPrice'])"
```

## 📞 Soporte y Recursos

### Recursos Adicionales
- **Documentación API**: http://localhost:8000/docs
- **Código Fuente**: https://github.com/mechmind-dwv/HelioBio-Economic
- **Ejemplos de Uso**: /notebooks/exploratory_analysis.ipynb

### Contacto para Soporte
- **Email**: ia.mechmind@gmail.com
- **GitHub Issues**: Reportar bugs y sugerencias
- **Documentación**: Consultar manuales avanzados

### Actualizaciones
- **Versión Actual**: v1.0.0
- **Próxima Versión**: v1.1.0 (ML Avanzado)
- **Frecuencia Updates**: Mensual

---

**🌞 ¡Gracias por usar HelioBio-Economic!**

*"Comprendiendo los ritmos del sol para anticipar los latidos de la economía"*
