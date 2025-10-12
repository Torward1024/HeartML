# /heart_model.py

from base_model import BaseModel
import pandas as pd

class HeartModel(BaseModel):
    """Конкретная модель для предсказания риска сердечного приступа"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.required_features = None
        
    def _validate_input(self, data: pd.DataFrame) -> bool:
        """Проверяет наличие всех необходимых исходных признаков"""
        if not self.is_loaded:
            self.load_model()
            
        # Получаем все исходные признаки из групп
        all_required = set()
        feature_groups = self.metadata.get('features_groups', {})
        for group_features in feature_groups.values():
            all_required.update(group_features)
        
        # Добавляем обязательные отдельные признаки
        individual_features = ['heart_rate', 'family_history', 'diet', 
                              'systolic_blood_pressure', 'diastolic_blood_pressure']
        all_required.update(individual_features)
        
        self.required_features = list(all_required)
        
        # Проверяем наличие всех признаков
        missing = set(self.required_features) - set(data.columns)
        if missing:
            raise ValueError(f"❌ Отсутствуют обязательные признаки: {missing}")
        
        return True
    
    def _preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Повторяет логику группировки признаков как при обучении"""
        df = data.copy()
        
        # Получаем пороги из метаданных
        thresholds = self.metadata.get('grouped_features_tresholds', {})
        feature_groups = self.metadata.get('features_groups', {})
        
        # 1. Создаем antrophometric_score
        if 'antrophometric_score' in feature_groups:
            bmi_thresh = thresholds.get('bmi', 0.5)
            age_thresh = thresholds.get('age', 0.4)
            
            df['antrophometric_score'] = (
                (df['bmi'] > bmi_thresh).astype(int) +
                df['obesity'] +
                (df['age'] > age_thresh).astype(int)
            )
        
        # 2. Создаем lifestyle_score
        if 'lifestyle_score' in feature_groups:
            stress_thresh = thresholds.get('stress_level', 7)
            ex_thresh = thresholds.get('exercise_hours_per_week', 0.5)
            sed_thresh = thresholds.get('sedentary_hours_per_day', 0.5)
            phys_thresh = thresholds.get('physical_activity_days_per_week', 3)
            sleep_thresh = thresholds.get('sleep_hours_per_day', 0.33)
            
            df['lifestyle_score'] = (
                df['smoking'] +
                (df['stress_level'] > stress_thresh).astype(int) +
                df['alcohol_consumption'] +
                (df['exercise_hours_per_week'] < ex_thresh).astype(int) +
                (df['sedentary_hours_per_day'] > sed_thresh).astype(int) +
                (df['physical_activity_days_per_week'] < phys_thresh).astype(int) +
                (df['sleep_hours_per_day'] < sleep_thresh).astype(int)
            )
        
        # 3. Создаем test_score
        if 'test_score' in feature_groups:
            chol_thresh = thresholds.get('cholesterol', 0.5)
            trig_thresh = thresholds.get('triglycerides', 0.5)
            sugar_thresh = thresholds.get('blood_sugar', 0.17)
            
            df['test_score'] = (
                (df['cholesterol'] > chol_thresh).astype(int) + 
                (df['triglycerides'] > trig_thresh).astype(int) + 
                (df['blood_sugar'] > sugar_thresh).astype(int)
            )
        
        # 4. Создаем medical_score
        if 'medical_score' in feature_groups:
            df['medical_score'] = (
                df['medication_use'] +
                df['diabetes'] +
                df['previous_heart_problems']
            )
        
        # 5. Выбираем только финальные признаки для модели
        final_features = self.metadata.get('feature_names', [])
        return df[final_features]
    
    def predict(self, csv_file_path: str) -> pd.DataFrame:
        """Предсказание для батча данных из CSV файла"""
        if not self.is_loaded:
            self.load_model()
        
        # Читаем CSV
        data = pd.read_csv(csv_file_path)
        
        # Сохраняем ID
        if 'id' not in data.columns:
            raise ValueError("❌ CSV файл должен содержать столбец 'id'")
        
        ids = data['id']
        
        # Удаляем ID из данных для предсказания
        data_for_prediction = data.drop('id', axis=1)
        
        # Валидация и препроцессинг
        self._validate_input(data_for_prediction)
        processed_data = self._preprocess_features(data_for_prediction)
        
        # Предсказание
        predictions = self.model.predict(processed_data)
        
        # Собираем результат
        result = pd.DataFrame({
            'id': ids,
            'prediction': predictions
        })
        
        return result
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_path: str):
        """Сохраняет предсказания в CSV файл"""
        predictions_df.to_csv(output_path, index=False)
        print(f"✅ Предсказания сохранены в {output_path}")