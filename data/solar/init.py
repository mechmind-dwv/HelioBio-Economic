"""
🌞 Datos Solares - HelioBio-Economic
Gestión de datos solares históricos y en tiempo real
"""

import pandas as pd
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta

SOLAR_DATA_PATH = os.path.dirname(__file__)

class SolarDataManager:
    """Gestor de datos solares"""
    
    def __init__(self):
        self.historical_file = os.path.join(SOLAR_DATA_PATH, 'historical_sunspots.csv')
        self.solar_cycles_file = os.path.join(SOLAR_DATA_PATH, 'solar_cycles.json')
        self.events_file = os.path.join(SOLAR_DATA_PATH, 'solar_events.parquet')
    
    def save_historical_data(self, data: pd.DataFrame) -> bool:
        """Guardar datos solares históricos"""
        try:
            data.to_csv(self.historical_file, index=True)
            return True
        except Exception as e:
            print(f"Error guardando datos solares: {e}")
            return False
    
    def load_historical_data(self) -> Optional[pd.DataFrame]:
        """Cargar datos solares históricos"""
        try:
            if os.path.exists(self.historical_file):
                return pd.read_csv(self.historical_file, index_col=0, parse_dates=True)
            return None
        except Exception as e:
            print(f"Error cargando datos solares: {e}")
            return None

# Instancia global
solar_data_manager = SolarDataManager()
