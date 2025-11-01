"""
🧠 solar_economic_ml.py
Machine Learning para Análisis de Correlaciones Solares-Económicas
Autor: Benjamin Cabeza Durán (mechmind-dwv)
Asistente: DeepSeek AI

Implementación de modelos ML avanzados para:
- Predicción de ciclos económicos basados en actividad solar
- Análisis de correlación no lineal
- Detección de patrones complejos solares-económicos
- Sistemas de alerta temprana
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow no disponible, usando modelos tradicionales")

# Series temporales
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """Configuración para modelos de Machine Learning"""
    model_type: str
    parameters: Dict[str, Any]
    feature_set: List[str]
    target_variable: str
    prediction_horizon: int
    cross_validation_folds: int = 5

@dataclass
class ModelPerformance:
    """Métricas de performance del modelo"""
    model_name: str
    r2_score: float
    mse: float
    mae: float
    rmse: float
    cross_val_mean: float
    cross_val_std: float
    feature_importance: Dict[str, float]
    training_time: float

@dataclass
class SolarEconomicPrediction:
    """Predicción solar-económica"""
    timestamp: datetime
    economic_indicator: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    solar_influence: float
    trend_direction: str
    anomaly_score: float

class SolarEconomicML:
    """
    Sistema de Machine Learning para análisis de correlaciones solares-económicas
    Implementa modelos avanzados para predicción y detección de patrones
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.model_performance = {}
        self.is_trained = False
        self.training_data = None
        
        # Configuración de modelos
        self.model_configs = {
            'random_forest_advanced': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'random_state': 42
                }
            },
            'poly_ridge': {
                'model': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                },
                'preprocessor': PolynomialFeatures(degree=2)
            },
            'xgboost_advanced': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': 150,
                    'max_depth': -1,
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'random_state': 42
                }
            }
        }
        
        if TF_AVAILABLE:
            self.model_configs['lstm_advanced'] = {
                'model_type': 'deep_learning',
                'params': {
                    'units': [50, 25, 10],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32
                }
            }
        
        # Variables objetivo para predicción
        self.target_variables = [
            'SP500',
            'GDP_growth', 
            'inflation_rate',
            'unemployment_rate',
            'consumer_confidence',
            'market_volatility'
        ]
        
        # Características solares
        self.solar_features = [
            'sunspot_number',
            'solar_flux',
            'geomagnetic_ap',
            'solar_wind_speed',
            'cme_activity',
            'solar_flares'
        ]
        
        # Características económicas
        self.economic_features = [
            'interest_rates',
            'money_supply',
            'industrial_production',
            'retail_sales',
            'housing_starts',
            'consumer_sentiment'
        ]
    
    async def train_models(self, economic_data: pd.DataFrame = None, 
                          solar_data: pd.DataFrame = None) -> Dict[str, ModelPerformance]:
        """
        Entrenar todos los modelos de ML con datos económicos y solares
        
        Args:
            economic_data: Datos económicos históricos
            solar_data: Datos solares históricos
            
        Returns:
            Diccionario con performance de cada modelo
        """
        logger.info("🧠 Iniciando entrenamiento de modelos ML...")
        
        if economic_data is None or solar_data is None:
            logger.warning("Datos no proporcionados, usando datos de ejemplo")
            economic_data, solar_data = await self._load_sample_data()
        
        try:
            # Preparar datos para entrenamiento
            X, y, feature_names = self._prepare_training_data(economic_data, solar_data)
            self.training_data = (X, y, feature_names)
            
            # Entrenar cada modelo
            for model_name, config in self.model_configs.items():
                logger.info(f"Entrenando modelo: {model_name}")
                
                if config.get('model_type') == 'deep_learning' and TF_AVAILABLE:
                    performance = await self._train_deep_learning_model(
                        model_name, X, y, config
                    )
                else:
                    performance = await self._train_traditional_model(
                        model_name, X, y, config, feature_names
                    )
                
                self.model_performance[model_name] = performance
            
            self.is_trained = True
            logger.info("✅ Todos los modelos entrenados correctamente")
            
            return self.model_performance
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento de modelos: {e}")
            raise
    
    async def _train_traditional_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                                     config: Dict, feature_names: List[str]) -> ModelPerformance:
        """Entrenar modelo tradicional (Random Forest, XGBoost, etc.)"""
        import time
        start_time = time.time()
        
        try:
            # Preprocesamiento de características
            if 'preprocessor' in config:
                preprocessor = config['preprocessor']
                X_processed = preprocessor.fit_transform(X)
            else:
                X_processed = X
            
            # Escalar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            self.scalers[model_name] = scaler
            
            # Crear y entrenar modelo
            model_class = config['model']
            model_params = config['params']
            model = model_class(**model_params)
            
            # Validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=config.get('cross_validation_folds', 5))
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                      scoring='r2', n_jobs=-1)
            
            # Entrenar modelo final
            model.fit(X_scaled, y)
            self.models[model_name] = model
            
            # Predicciones y métricas
            y_pred = model.predict(X_scaled)
            
            # Calcular importancia de características
            feature_importance = self._calculate_feature_importance(
                model, feature_names, model_name
            )
            self.feature_importances[model_name] = feature_importance
            
            # Métricas de performance
            training_time = time.time() - start_time
            performance = ModelPerformance(
                model_name=model_name,
                r2_score=r2_score(y, y_pred),
                mse=mean_squared_error(y, y_pred),
                mae=mean_absolute_error(y, y_pred),
                rmse=np.sqrt(mean_squared_error(y, y_pred)),
                cross_val_mean=np.mean(cv_scores),
                cross_val_std=np.std(cv_scores),
                feature_importance=feature_importance,
                training_time=training_time
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {model_name}: {e}")
            # Retornar performance por defecto en caso de error
            return ModelPerformance(
                model_name=model_name,
                r2_score=-1.0,
                mse=float('inf'),
                mae=float('inf'),
                rmse=float('inf'),
                cross_val_mean=-1.0,
                cross_val_std=0.0,
                feature_importance={},
                training_time=0.0
            )
    
    async def _train_deep_learning_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                                       config: Dict) -> ModelPerformance:
        """Entrenar modelo de Deep Learning (LSTM)"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no disponible para modelos DL")
        
        import time
        start_time = time.time()
        
        try:
            # Preparar datos para LSTM
            X_reshaped = self._prepare_lstm_data(X)
            y_reshaped = y.reshape(-1, 1)
            
            # Crear modelo LSTM
            model = self._create_lstm_model(
                input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]),
                units=config['params']['units'],
                dropout_rate=config['params']['dropout_rate']
            )
            
            # Compilar modelo
            model.compile(
                optimizer=Adam(learning_rate=config['params']['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            # Entrenar modelo
            history = model.fit(
                X_reshaped, y_reshaped,
                epochs=config['params']['epochs'],
                batch_size=config['params']['batch_size'],
                validation_split=0.2,
                verbose=0
            )
            
            self.models[model_name] = model
            
            # Predicciones y métricas
            y_pred = model.predict(X_reshaped).flatten()
            
            training_time = time.time() - start_time
            performance = ModelPerformance(
                model_name=model_name,
                r2_score=r2_score(y, y_pred),
                mse=mean_squared_error(y, y_pred),
                mae=mean_absolute_error(y, y_pred),
                rmse=np.sqrt(mean_squared_error(y, y_pred)),
                cross_val_mean=0.0,  # No CV para DL por simplicidad
                cross_val_std=0.0,
                feature_importance={},  # DL no tiene importancia directa
                training_time=training_time
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error entrenando modelo DL {model_name}: {e}")
            raise
    
    def _prepare_lstm_data(self, X: np.ndarray, time_steps: int = 10) -> np.ndarray:
        """Preparar datos para modelo LSTM"""
        X_reshaped = []
        for i in range(time_steps, len(X)):
            X_reshaped.append(X[i-time_steps:i])
        return np.array(X_reshaped)
    
    def _create_lstm_model(self, input_shape: Tuple[int, int], 
                          units: List[int], dropout_rate: float) -> tf.keras.Model:
        """Crear modelo LSTM avanzado"""
        model = Sequential()
        
        # Capa LSTM inicial
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Capas LSTM adicionales
        for i, unit in enumerate(units[1:-1]):
            model.add(LSTM(unit, return_sequences=(i < len(units)-2)))
            model.add(Dropout(dropout_rate))
        
        # Capa final
        model.add(Dense(units[-1], activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        return model
    
    def _prepare_training_data(self, economic_data: pd.DataFrame, 
                             solar_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preparar datos para entrenamiento de modelos"""
        logger.info("📊 Preparando datos para entrenamiento ML...")
        
        try:
            # Combinar datos económicos y solares
            combined_data = self._merge_economic_solar_data(economic_data, solar_data)
            
            # Crear características de ingeniería
            engineered_features = self._create_engineered_features(combined_data)
            
            # Seleccionar características y objetivo
            X, y, feature_names = self._select_features_and_target(engineered_features)
            
            logger.info(f"✅ Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
            return X, y, feature_names
            
        except Exception as e:
            logger.error(f"Error preparando datos de entrenamiento: {e}")
            raise
    
    def _merge_economic_solar_data(self, economic_data: pd.DataFrame, 
                                 solar_data: pd.DataFrame) -> pd.DataFrame:
        """Combinar datos económicos y solares"""
        # Alinear por fecha
        economic_data = economic_data.copy()
        solar_data = solar_data.copy()
        
        economic_data.index = pd.to_datetime(economic_data.index)
        solar_data.index = pd.to_datetime(solar_data.index)
        
        # Combinar datos
        combined_data = pd.merge(economic_data, solar_data, 
                               left_index=True, right_index=True, how='inner')
        
        # Llenar valores faltantes
        combined_data = combined_data.ffill().bfill()
        
        return combined_data
    
    def _create_engineered_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crear características de ingeniería avanzadas"""
        engineered_data = data.copy()
        
        # Características temporales
        engineered_data['day_of_year'] = engineered_data.index.dayofyear
        engineered_data['month'] = engineered_data.index.month
        engineered_data['quarter'] = engineered_data.index.quarter
        engineered_data['year'] = engineered_data.index.year
        
        # Lags para características solares
        solar_cols = [col for col in data.columns if any(solar_feat in col for solar_feat in self.solar_features)]
        for col in solar_cols:
            for lag in [1, 3, 6, 12]:  # Lags en meses
                engineered_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Lags para características económicas
        economic_cols = [col for col in data.columns if any(econ_feat in col for econ_feat in self.economic_features)]
        for col in economic_cols:
            for lag in [1, 3, 6]:
                engineered_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Medias móviles
        for col in solar_cols + economic_cols:
            engineered_data[f'{col}_ma_3'] = data[col].rolling(3).mean()
            engineered_data[f'{col}_ma_6'] = data[col].rolling(6).mean()
            engineered_data[f'{col}_ma_12'] = data[col].rolling(12).mean()
        
        # Diferencias estacionales
        for col in economic_cols:
            engineered_data[f'{col}_diff_1'] = data[col].diff()
            engineered_data[f'{col}_diff_12'] = data[col].diff(12)
        
        # Interacciones solares-económicas
        for solar_col in solar_cols[:3]:  # Primeras 3 características solares
            for economic_col in economic_cols[:3]:  # Primeras 3 económicas
                engineered_data[f'{solar_col}_{economic_col}_interaction'] = (
                    data[solar_col] * data[economic_col]
                )
        
        # Ratios solares-económicos
        for solar_col in solar_cols[:2]:
            for economic_col in economic_cols[:2]:
                engineered_data[f'{solar_col}_{economic_col}_ratio'] = (
                    data[solar_col] / (data[economic_col] + 1e-8)  # Evitar división por cero
                )
        
        # Eliminar filas con NaN resultantes de las transformaciones
        engineered_data = engineered_data.dropna()
        
        return engineered_data
    
    def _select_features_and_target(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Seleccionar características y variable objetivo"""
        # Excluir columnas no numéricas y de fecha
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Seleccionar características (excluir objetivos)
        feature_cols = [col for col in numeric_cols if not any(target in col for target in self.target_variables)]
        
        # Usar SP500 como objetivo principal (puede extenderse)
        target_col = 'SP500' if 'SP500' in data.columns else numeric_cols[0]
        
        X = data[feature_cols].values
        y = data[target_col].values
        
        return X, y, feature_cols
    
    def _calculate_feature_importance(self, model, feature_names: List[str], 
                                    model_name: str) -> Dict[str, float]:
        """Calcular importancia de características"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                # Para modelos sin importancia directa
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            # Normalizar importancias
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)
            
            # Crear diccionario ordenado
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return sorted_importance
            
        except Exception as e:
            logger.warning(f"No se pudo calcular importancia para {model_name}: {e}")
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    async def predict_economic_cycles(self, cycles: List[str], 
                                    horizon_days: int = 30) -> Dict[str, Any]:
        """
        Predecir ciclos económicos basados en factores solares
        
        Args:
            cycles: Lista de ciclos a incluir
            horizon_days: Horizonte de predicción en días
            
        Returns:
            Predicciones y análisis
        """
        if not self.is_trained:
            raise ValueError("Modelos no entrenados. Ejecutar train_models() primero.")
        
        logger.info(f"🎯 Prediciendo ciclos económicos - Horizonte: {horizon_days} días")
        
        try:
            predictions = {}
            
            for model_name, model in self.models.items():
                if model_name in self.model_performance:
                    # Preparar datos para predicción
                    X_pred = self._prepare_prediction_data(horizon_days)
                    
                    # Realizar predicción
                    if model_name.startswith('lstm'):
                        # Predicción para LSTM
                        X_reshaped = self._prepare_lstm_data(X_pred)
                        y_pred = model.predict(X_reshaped).flatten()
                    else:
                        # Predicción para modelos tradicionales
                        if 'preprocessor' in self.model_configs[model_name]:
                            preprocessor = self.model_configs[model_name]['preprocessor']
                            X_processed = preprocessor.transform(X_pred)
                        else:
                            X_processed = X_pred
                        
                        X_scaled = self.scalers[model_name].transform(X_processed)
                        y_pred = model.predict(X_scaled)
                    
                    # Calcular intervalos de confianza
                    confidence_interval = self._calculate_confidence_interval(
                        y_pred, model_name
                    )
                    
                    predictions[model_name] = {
                        'predictions': y_pred.tolist(),
                        'confidence_interval': confidence_interval,
                        'model_performance': self.model_performance[model_name].__dict__,
                        'solar_influence': self._calculate_solar_influence(model_name),
                        'trend': self._analyze_prediction_trend(y_pred)
                    }
            
            # Combinar predicciones de todos los modelos (ensamble)
            ensemble_prediction = self._create_ensemble_prediction(predictions)
            
            return {
                'individual_predictions': predictions,
                'ensemble_prediction': ensemble_prediction,
                'horizon_days': horizon_days,
                'prediction_timestamp': datetime.now().isoformat(),
                'model_consensus': self._analyze_model_consensus(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error en predicción de ciclos económicos: {e}")
            raise
    
    def _prepare_prediction_data(self, horizon_days: int) -> np.ndarray:
        """Preparar datos para predicción futura"""
        # En implementación real, esto usaría datos actuales y proyecciones
        # Por ahora, usamos los últimos datos de entrenamiento
        if self.training_data is None:
            raise ValueError("No hay datos de entrenamiento disponibles")
        
        X, _, _ = self.training_data
        return X[-horizon_days:]  # Usar últimos datos para predicción
    
    def _calculate_confidence_interval(self, predictions: np.ndarray, 
                                     model_name: str) -> Tuple[float, float]:
        """Calcular intervalo de confianza para predicciones"""
        # Basado en performance del modelo y variabilidad de predicciones
        model_perf = self.model_performance[model_name]
        std_dev = np.std(predictions)
        
        confidence_level = 0.95
        z_score = 1.96  # Para 95% de confianza
        
        margin_of_error = z_score * std_dev * np.sqrt(1 + 1/len(predictions))
        
        mean_prediction = np.mean(predictions)
        lower_bound = mean_prediction - margin_of_error
        upper_bound = mean_prediction + margin_of_error
        
        return (float(lower_bound), float(upper_bound))
    
    def _calculate_solar_influence(self, model_name: str) -> float:
        """Calcular influencia solar en las predicciones"""
        if model_name in self.feature_importances:
            solar_features = [f for f in self.feature_importances[model_name].keys() 
                            if any(solar_feat in f for solar_feat in self.solar_features)]
            
            if solar_features:
                total_importance = sum(self.feature_importances[model_name].values())
                solar_importance = sum(self.feature_importances[model_name][f] 
                                     for f in solar_features)
                return solar_importance / total_importance
        
        return 0.0
    
    def _analyze_prediction_trend(self, predictions: np.ndarray) -> str:
        """Analizar tendencia de las predicciones"""
        if len(predictions) < 2:
            return "neutral"
        
        # Regresión lineal simple para determinar tendencia
        x = np.arange(len(predictions))
        slope = np.polyfit(x, predictions, 1)[0]
        
        if slope > 0.01:
            return "bullish"
        elif slope < -0.01:
            return "bearish"
        else:
            return "neutral"
    
    def _create_ensemble_prediction(self, individual_predictions: Dict) -> Dict[str, Any]:
        """Crear predicción de ensamble combinando todos los modelos"""
        all_predictions = []
        
        for model_pred in individual_predictions.values():
            all_predictions.append(model_pred['predictions'])
        
        # Promedio ponderado por performance del modelo
        ensemble_pred = np.zeros_like(all_predictions[0])
        total_weight = 0
        
        for model_name, model_pred in individual_predictions.items():
            weight = max(0, self.model_performance[model_name].r2_score)
            ensemble_pred += np.array(model_pred['predictions']) * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return {
            'predictions': ensemble_pred.tolist(),
            'ensemble_method': 'weighted_average_by_r2',
            'models_combined': len(individual_predictions)
        }
    
    def _analyze_model_consensus(self, predictions: Dict) -> Dict[str, Any]:
        """Analizar consenso entre diferentes modelos"""
        directions = []
        for model_pred in predictions.values():
            directions.append(model_pred['trend'])
        
        bull_count = directions.count('bullish')
        bear_count = directions.count('bearish')
        neutral_count = directions.count('neutral')
        
        total_models = len(directions)
        
        return {
            'bullish_consensus': bull_count / total_models,
            'bearish_consensus': bear_count / total_models,
            'neutral_consensus': neutral_count / total_models,
            'dominant_trend': max(set(directions), key=directions.count),
            'agreement_level': max(bull_count, bear_count, neutral_count) / total_models
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Obtener métricas de performance de todos los modelos"""
        performance_summary = {}
        
        for model_name, perf in self.model_performance.items():
            performance_summary[model_name] = {
                'r2_score': perf.r2_score,
                'rmse': perf.rmse,
                'mae': perf.mae,
                'cross_validation_mean': perf.cross_val_mean,
                'cross_validation_std': perf.cross_val_std,
                'training_time_seconds': perf.training_time,
                'solar_feature_influence': self._calculate_solar_influence(model_name)
            }
        
        # Métricas agregadas
        performance_summary['aggregate'] = {
            'best_model': max(self.model_performance.items(), 
                            key=lambda x: x[1].r2_score)[0],
            'average_r2': np.mean([p.r2_score for p in self.model_performance.values()]),
            'model_count': len(self.model_performance),
            'training_status': self.is_trained
        }
        
        return performance_summary
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Obtener importancia de características agregada"""
        if not self.feature_importances:
            return {}
        
        # Combinar importancia de todos los modelos
        combined_importance = {}
        total_models = len(self.feature_importances)
        
        for model_name, importance_dict in self.feature_importances.items():
            for feature, importance in importance_dict.items():
                if feature in combined_importance:
                    combined_importance[feature] += importance
                else:
                    combined_importance[feature] = importance
        
        # Promediar y ordenar
        for feature in combined_importance:
            combined_importance[feature] /= total_models
        
        sorted_importance = dict(sorted(
            combined_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Separar características solares y económicas
        solar_features = {k: v for k, v in sorted_importance.items() 
                         if any(sf in k for sf in self.solar_features)}
        economic_features = {k: v for k, v in sorted_importance.items() 
                           if any(ef in k for ef in self.economic_features)}
        other_features = {k: v for k, v in sorted_importance.items() 
                         if k not in solar_features and k not in economic_features}
        
        return {
            'overall_importance': sorted_importance,
            'solar_features': solar_features,
            'economic_features': economic_features,
            'other_features': other_features,
            'solar_economic_ratio': (
                sum(solar_features.values()) / 
                (sum(economic_features.values()) + 1e-8)
            )
        }
    
    async def _load_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cargar datos de ejemplo para desarrollo"""
        logger.warning("📝 Cargando datos de ejemplo para desarrollo")
        
        # Datos económicos de ejemplo
        dates = pd.date_range('2000-01-01', '2023-12-31', freq='M')
        economic_data = pd.DataFrame({
            'SP500': np.cumsum(np.random.normal(0, 1, len(dates))) + 1000,
            'GDP_growth': np.random.normal(2, 0.5, len(dates)),
            'inflation_rate': np.random.normal(2, 0.3, len(dates)),
            'unemployment_rate': np.random.normal(5, 1, len(dates)),
            'interest_rates': np.random.normal(3, 1, len(dates))
        }, index=dates)
        
        # Datos solares de ejemplo con patrones cíclicos
        solar_data = pd.DataFrame({
            'sunspot_number': 50 + 40 * np.sin(2 * np.pi * np.arange(len(dates)) / 132) + np.random.normal(0, 10, len(dates)),
            'solar_flux': 70 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 132) + np.random.normal(0, 5, len(dates)),
            'geomagnetic_ap': np.random.normal(10, 3, len(dates)),
            'solar_wind_speed': 400 + 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 132) + np.random.normal(0, 20, len(dates))
        }, index=dates)
        
        return economic_data, solar_data

# Instancia global para uso en otros módulos
solar_economic_ml = SolarEconomicML()
