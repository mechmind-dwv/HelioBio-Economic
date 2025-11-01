#!/bin/bash

echo "🚀 Instalando HelioBio-Economic v1.0"
echo "======================================"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no encontrado. Por favor instala Python 3.8+"
    exit 1
fi

# Crear entorno virtual
echo "📁 Creando entorno virtual..."
python3 -m venv helio_env

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source helio_env/bin/activate

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Crear directorios necesarios
echo "📁 Creando estructura de directorios..."
mkdir -p logs
mkdir -p data/solar
mkdir -p data/economic
mkdir -p data/models
mkdir -p data/cache
mkdir -p app/static

# Crear archivo de configuración
echo "⚙️ Creando archivo de configuración..."
cat > .env << EOL
# NASA API Configuration
NASA_API_KEY=DEMO_KEY

# Economic Data APIs
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
EOL

echo "✅ Instalación completada!"
echo ""
echo "📝 Próximos pasos:"
echo "1. Edita el archivo .env con tus claves API"
echo "2. Ejecuta: source helio_env/bin/activate"
echo "3. Inicia la aplicación: python app/main.py"
echo "4. Abre http://localhost:8000 en tu navegador"
echo ""
echo "🌞 ¡HelioBio-Economic está listo!"
