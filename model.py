from abc import ABC, abstractmethod
import pandas as pd
import joblib
import numpy as np

class BaseMedicalModel(ABC):
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
            print("✅ Модель с метаданными загружена")
        except Exception as e:
            raise Exception(f"❌ Ошибка загрузки модели: {e}")
    
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
    
    def predict(self, input_data) -> dict:
        """Основной метод для предсказания"""
        if not self.is_loaded:
            self.load_model()
        
        # Конвертируем в DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Валидация
        if not self._validate_input(df):
            raise ValueError("❌ Невалидные входные данные")
        
        # Препроцессинг
        processed_data = self._preprocess_features(df)
        
        # Предсказание
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        return self._format_output(prediction, probability)
    
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