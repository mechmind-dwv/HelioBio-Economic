"""
💹 Datos Económicos - HelioBio-Economic
Gestión de datos económicos y financieros
"""

import pandas as pd
import os
import json
from typing import Dict, List, Optional
from datetime import datetime

ECONOMIC_DATA_PATH = os.path.dirname(__file__)

class EconomicDataManager:
    """Gestor de datos económicos"""
    
    def __init__(self):
        self.market_data_file = os.path.join(ECONOMIC_DATA_PATH, 'market_data.parquet')
        self.indicators_file = os.path.join(ECONOMIC_DATA_PATH, 'economic_indicators.json')
        self.crisis_data_file = os.path.join(ECONOMIC_DATA_PATH, 'historical_crises.csv')
    
    def save_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Guardar datos de mercado"""
        try:
            file_path = os.path.join(ECONOMIC_DATA_PATH, f'market_{symbol}.parquet')
            data.to_parquet(file_path)
            return True
        except Exception as e:
            print(f"Error guardando datos de mercado {symbol}: {e}")
            return False
    
    def load_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Cargar datos de mercado"""
        try:
            file_path = os.path.join(ECONOMIC_DATA_PATH, f'market_{symbol}.parquet')
            if os.path.exists(file_path):
                return pd.read_parquet(file_path)
            return None
        except Exception as e:
            print(f"Error cargando datos de mercado {symbol}: {e}")
            return None
    
    def save_economic_indicators(self, indicators: Dict) -> bool:
        """Guardar indicadores económicos"""
        try:
            with open(self.indicators_file, 'w') as f:
                json.dump(indicators, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error guardando indicadores económicos: {e}")
            return False
    
    def load_economic_indicators(self) -> Optional[Dict]:
        """Cargar indicadores económicos"""
        try:
            if os.path.exists(self.indicators_file):
                with open(self.indicators_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error cargando indicadores económicos: {e}")
            return None

# Instancia global
economic_data_manager = EconomicDataManager()
