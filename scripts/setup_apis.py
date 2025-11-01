#!/usr/bin/env python3
"""
🔧 Script de Configuración de APIs - HelioBio-Economic
Configuración interactiva de claves API y servicios externos
"""

import os
import getpass
from dotenv import load_dotenv, set_key

def setup_apis_interactive():
    """Configuración interactiva de APIs"""
    print("🌞 Configuración de APIs - HelioBio-Economic")
    print("=" * 50)
    
    # Cargar configuración existente
    load_dotenv()
    
    config = {}
    
    # NASA API
    print("\n1. 🌌 NASA DONKI API")
    print("   Obtén tu API key en: https://api.nasa.gov/")
    nasa_key = getpass.getpass("   NASA API Key [actual: {}]: ".format(
        os.getenv('NASA_API_KEY', 'DEMO_KEY')
    )) or os.getenv('NASA_API_KEY', 'DEMO_KEY')
    config['NASA_API_KEY'] = nasa_key
    
    # FRED API
    print("\n2. 💹 FRED API (Federal Reserve Economic Data)")
    print("   Obtén tu API key en: https://fred.stlouisfed.org/docs/api/api_key.html")
    fred_key = getpass.getpass("   FRED API Key [actual: {}]: ".format(
        os.getenv('FRED_API_KEY', '')
    )) or os.getenv('FRED_API_KEY', '')
    if fred_key:
        config['FRED_API_KEY'] = fred_key
    
    # Alpha Vantage
    print("\n3. 📊 Alpha Vantage API")
    print("   Obtén tu API key en: https://www.alphavantage.org/support/#api-key")
    alpha_key = getpass.getpass("   Alpha Vantage Key [actual: {}]: ".format(
        os.getenv('ALPHA_VANTAGE_KEY', '')
    )) or os.getenv('ALPHA_VANTAGE_KEY', '')
    if alpha_key:
        config['ALPHA_VANTAGE_KEY'] = alpha_key
    
    # Guardar configuración
    env_file = '.env'
    for key, value in config.items():
        set_key(env_file, key, value)
    
    print(f"\n✅ Configuración guardada en {env_file}")
    print("\n📋 Resumen de configuración:")
    for key, value in config.items():
        masked_value = value[:8] + '***' if len(value) > 8 else '***'
        print(f"   {key}: {masked_value}")
    
    print("\n🎯 ¡Configuración completada!")
    print("   Ejecuta: python app/main.py para iniciar la aplicación")

if __name__ == "__main__":
    setup_apis_interactive()
