"""
💾 Sistema de Caché - HelioBio-Economic
Gestión de caché para optimizar rendimiento
"""

import pickle
import os
import hashlib
from typing import Any, Optional
from datetime import datetime, timedelta

CACHE_PATH = os.path.dirname(__file__)

class CacheManager:
    """Gestor de caché distribuido"""
    
    def __init__(self, default_ttl: int = 3600):  # 1 hora por defecto
        self.cache_dir = CACHE_PATH
        self.default_ttl = default_ttl
        
    def _get_cache_key(self, key: str) -> str:
        """Generar clave de caché única"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Obtener ruta de archivo de caché"""
        cache_key = self._get_cache_key(key)
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Guardar valor en caché"""
        try:
            cache_path = self._get_cache_path(key)
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            cache_data = {
                'value': value,
                'expires_at': expires_at,
                'created_at': datetime.now()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            print(f"Error guardando en caché: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor de caché"""
        try:
            cache_path = self._get_cache_path(key)
            
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verificar expiración
            if datetime.now() > cache_data['expires_at']:
                self.delete(key)
                return None
            
            return cache_data['value']
        except Exception as e:
            print(f"Error obteniendo de caché: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Eliminar valor de caché"""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return True
        except Exception as e:
            print(f"Error eliminando de caché: {e}")
            return False
    
    def clear_expired(self) -> int:
        """Limpiar entradas expiradas y retornar count"""
        try:
            expired_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            cache_data = pickle.load(f)
                        
                        if datetime.now() > cache_data['expires_at']:
                            os.remove(filepath)
                            expired_count += 1
                    except:
                        # Si hay error al leer, eliminar archivo corrupto
                        os.remove(filepath)
                        expired_count += 1
            
            return expired_count
        except Exception as e:
            print(f"Error limpiando caché expirado: {e}")
            return 0

# Instancia global
cache_manager = CacheManager()
