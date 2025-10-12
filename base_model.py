# /base_model.py

from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import joblib

class BaseModel(ABC):
    """Абстрактный базовый класс для всех медицинских моделей"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.is_loaded = False
        
    def load_model(self):
        """Загружает модель с метаданными"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']              # получаем саму модель
            self.metadata = model_data['metadata']        # получаем метаданные
            self.is_loaded = True
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def get_model_info(self) -> dict:
        """Возвращает информацию о модели"""
        if not self.is_loaded:
            self.load_model()
            
        return {
            'model_type': self.metadata.get('model_type', 'unknown'),
            'features_used': self.metadata.get('feature_names', []),
            'features_groups': self.metadata.get('features_groups', {}),
            'grouped_features_tresholds': self.metadata.get('grouped_features_tresholds', {}),
            'drop_features': self.metadata.get('drop_features', []),
            'target_feature': self.metadata.get('target_feature', 'unknown'),
            'threshold': self.metadata.get('threshold', 0.25)
        }
    
    def predict(self, input_data: Union[dict, pd.DataFrame]) -> dict:
        """Основной метод для предсказания"""
        try:
            if not self.is_loaded:
                self.load_model()
        
            # convert dict ot pandas df
            if isinstance(input_data, dict):
                data = pd.DataFrame([input_data])
            else:
                data = input_data.copy()
        
            # need to validate data
            if not self._validate_input(data):
                raise ValueError("Invalid data!")
        
            # perform data preprocessing
            processed_data = self._preprocess_features(data)
        
            pred = self.model.predict(processed_data)
            proba = self.model.predict_proba(processed_data)
            status = 'OK'
            return status, self._format_output(pred, proba)
        except:
            status = 'Failed'
            return status, {}
    
    @abstractmethod
    def _preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Абстрактный метод для препроцессинга фич"""
        pass
    
    @abstractmethod
    def _validate_input(self, data: pd.DataFrame) -> bool:
        """Абстрактный метод для валидации входных данных"""
        pass

    @abstractmethod
    def _format_output(self, prediction, probability) -> dict:
        """Форматирует выходные данные"""
        pass    