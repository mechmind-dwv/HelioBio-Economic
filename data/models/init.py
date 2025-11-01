"""
🧠 Modelos ML - HelioBio-Economic
Gestión de modelos de Machine Learning entrenados
"""

import pickle
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

MODELS_PATH = os.path.dirname(__file__)

class ModelManager:
    """Gestor de modelos ML"""
    
    def __init__(self):
        self.models_dir = MODELS_PATH
        self.metadata_file = os.path.join(MODELS_PATH, 'models_metadata.json')
        
    def save_model(self, model_name: str, model_object: Any, 
                  performance: Dict[str, Any], features: List[str]) -> bool:
        """Guardar modelo entrenado"""
        try:
            # Guardar modelo
            model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)
            
            # Actualizar metadatos
            metadata = self.load_metadata()
            metadata[model_name] = {
                'name': model_name,
                'trained_at': datetime.now().isoformat(),
                'performance': performance,
                'features': features,
                'file_path': model_path
            }
            
            self.save_metadata(metadata)
            return True
            
        except Exception as e:
            print(f"Error guardando modelo {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Cargar modelo entrenado"""
        try:
            model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            print(f"Error cargando modelo {model_name}: {e}")
            return None
    
    def load_metadata(self) -> Dict[str, Any]:
        """Cargar metadatos de modelos"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error cargando metadatos: {e}")
            return {}
    
    def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Guardar metadatos de modelos"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            print(f"Error guardando metadatos: {e}")
            return False
    
    def get_best_model(self) -> Optional[str]:
        """Obtener el mejor modelo basado en métricas"""
        metadata = self.load_metadata()
        if not metadata:
            return None
        
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, model_data in metadata.items():
            r2 = model_data.get('performance', {}).get('r2_score', -1)
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name
        
        return best_model

# Instancia global
model_manager = ModelManager()
