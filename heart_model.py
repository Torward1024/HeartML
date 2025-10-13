# /heart_model.py

from base_model import BaseModel
from typing import Union
import re
import pandas as pd

class HeartModel(BaseModel):
    """Model to predict heart attack risk binary."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.load_model()
        self.threshold = self.metadata.get('threshold', 0.25)

    def _to_snake_case(self, name: str):
        """Method to rename columns in DataFrame"""
        name = re.sub(r'[\s\-()]', '_', name)
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        name = name.lower()
        name = re.sub(r'_+', '_', name)
        return name.strip('_')
        
    def _validate_input(self, data: pd.DataFrame) -> bool:
        """Validate data before prediction"""
        if not self.is_loaded:
            self.load_model()

        # rename columns
        data.columns = [self._to_snake_case(col) for col in data.columns]

        if 'id' not in data.columns:
            raise ValueError("CSV must contain 'id' column")
        
        self.ids = data['id']
        self.data_for_prediction = data.drop('id', axis=1)
            
        # get required features from model metadata
        all_required = set()
        feature_groups = self.metadata.get('features_groups', {})
        for group_features in feature_groups.values():
            all_required.update(group_features)
        
        # add additional individual features
        individual_features = ['heart_rate', 'family_history', 'diet', 
                              'systolic_blood_pressure', 'diastolic_blood_pressure']
        all_required.update(individual_features)
        
        self.required_features = list(all_required)
        
        # check all features
        missing = set(self.required_features) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        if self.data_for_prediction.isna().any().any():
            # numeric columns -- fill with median
            numeric_cols = self.data_for_prediction.select_dtypes(include=['float64', 'int64']).columns
            self.data_for_prediction[numeric_cols] = \
            self.data_for_prediction[numeric_cols].fillna(self.data_for_prediction[numeric_cols].median())
            
            # сategorical columns -- fill with a default value 'unknown'
            categorical_cols = self.data_for_prediction.select_dtypes(include=['object']).columns
            self.data_for_prediction[categorical_cols] = \
            self.data_for_prediction[categorical_cols].fillna('unknown')
        
        return True
    
    def _preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Method to preprocess data according to the requirements achieved in the research."""
        data = data.copy()
        thresholds = self.metadata.get('grouped_features_tresholds', {})
        feature_groups = self.metadata.get('features_groups', {})
        if not feature_groups:
            raise ValueError("Metadata must contain 'features_groups'")
        if not thresholds:
            raise ValueError("Metadata must contain 'grouped_features_tresholds'")
        
        # grouped feature `antrophometric_score`
        if 'antrophometric_score' in feature_groups:
            bmi_thresh = thresholds.get('bmi', 0.5)
            age_thresh = thresholds.get('age', 0.4)
            
            data['antrophometric_score'] = (
                (data['bmi'] > bmi_thresh).astype(int) +
                data['obesity'] +
                (data['age'] > age_thresh).astype(int)
            )
        # grouped feature `lifestyle_score`
        if 'lifestyle_score' in feature_groups:
            stress_thresh = thresholds.get('stress_level', 7)
            ex_thresh = thresholds.get('exercise_hours_per_week', 0.5)
            sed_thresh = thresholds.get('sedentary_hours_per_day', 0.5)
            phys_thresh = thresholds.get('physical_activity_days_per_week', 3)
            sleep_thresh = thresholds.get('sleep_hours_per_day', 0.33)
            
            data['lifestyle_score'] = (
                data['smoking'] +
                (data['stress_level'] > stress_thresh).astype(int) +
                data['alcohol_consumption'] +
                (data['exercise_hours_per_week'] < ex_thresh).astype(int) +
                (data['sedentary_hours_per_day'] > sed_thresh).astype(int) +
                (data['physical_activity_days_per_week'] < phys_thresh).astype(int) +
                (data['sleep_hours_per_day'] < sleep_thresh).astype(int)
            )
        
        # grouped feature `test_score`
        if 'test_score' in feature_groups:
            chol_thresh = thresholds.get('cholesterol', 0.5)
            trig_thresh = thresholds.get('triglycerides', 0.5)
            sugar_thresh = thresholds.get('blood_sugar', 0.17)
            
            data['test_score'] = (
                (data['cholesterol'] > chol_thresh).astype(int) + 
                (data['triglycerides'] > trig_thresh).astype(int) + 
                (data['blood_sugar'] > sugar_thresh).astype(int)
            )
        
        # grouped feature `medical_score`
        if 'medical_score' in feature_groups:
            data['medical_score'] = (
                data['medication_use'] +
                data['diabetes'] +
                data['previous_heart_problems']
            )
        
        # 
        final_features = self.metadata.get('feature_names', [])
        return data[final_features]
    
    def _format_output(self, prediction, probability) -> pd.DataFrame:
        """Prepare the data for output"""        
        if hasattr(probability, 'shape') and len(probability.shape) > 1:
            probability = probability[:, 1]
        
        result = pd.DataFrame({
            'id': self.ids,
            'prediction': prediction
        })
        return result
    
    def get_treshold(self) -> float:
        """Get the treshold from model"""
        return self.threshold
    
    def set_treshold(self, threshold: int) -> float:
        """Sets new treshold to the model"""
        self.threshold = threshold
        return self.threshold
    
    def predict(self, input_data: Union[dict, pd.DataFrame]) -> dict:
        """Predict heart attack risk with custom threshold -- overloaded from BaseModel"""
        try:
            if not self.is_loaded:
                self.load_model()
        
            # convert dict to pandas df
            if isinstance(input_data, dict):
                data = pd.DataFrame([input_data])
            else:
                data = input_data.copy()
        
            # validate data
            if not self._validate_input(data):
                raise ValueError("Invalid data!")
        
            # perform data preprocessing
            processed_data = self._preprocess_features(data)

            # get probabilities and apply custom threshold
            proba = self.model.predict_proba(processed_data)[:, 1]
            pred = (proba > self.threshold).astype(int)
            return 'success', self._format_output(pred, proba)
        except Exception as e:
            return 'error', str(e)