import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, process_result

class PredictPipeline:
  def __init__(self):
    pass

  def predict(self, features):
    try:
      model_path = 'artifacts/model.joblib'
      preprocessor_path = 'artifacts/preprocessor.joblib'
      model = load_object(file_path = model_path)
      preprocessor = load_object(file_path=preprocessor_path)
      data_scaled = preprocessor.transform(features)
      preds = model.predict(data_scaled)
      result = process_result(preds)
      return result
    
    except Exception as e:
      raise CustomException(e, sys)
    

class CustomData:
  def __init__(self,
              baseline_value: float,
              accelerations: float,
              fetal_movement: float,
              uterine_contractions: float,
              light_decelerations: float,
              severe_decelerations: float,
              prolongued_decelerations: float,
              abnormal_short_term_variability: float,
              mean_value_of_short_term_variability: float,
              percentage_of_time_with_abnormal_long_term_variability: float,
              mean_value_of_long_term_variability: float,
              histogram_width: float,
              histogram_min: float,
              histogram_max: float,
              histogram_number_of_peaks: float,
              histogram_number_of_zeroes: float,
              histogram_mode: float,
              histogram_mean: float,
              histogram_median: float,
              histogram_variance: float,
              histogram_tendency: float ):
    self.baseline_value = baseline_value
    self.accelerations = accelerations
    self.fetal_movement = fetal_movement
    self.uterine_contractions = uterine_contractions
    self.light_decelerations = light_decelerations
    self.severe_decelerations = severe_decelerations
    self.prolongued_decelerations = prolongued_decelerations
    self.abnormal_short_term_variability = abnormal_short_term_variability
    self.mean_value_of_short_term_variability = mean_value_of_short_term_variability
    self.percentage_of_time_with_abnormal_long_term_variability = percentage_of_time_with_abnormal_long_term_variability
    self.mean_value_of_long_term_variability = mean_value_of_long_term_variability
    self.histogram_width = histogram_width
    self.histogram_min = histogram_min
    self.histogram_max = histogram_max
    self.histogram_number_of_peaks = histogram_number_of_peaks
    self.histogram_number_of_zeroes = histogram_number_of_zeroes
    self.histogram_mode = histogram_mode
    self.histogram_mean = histogram_mean
    self.histogram_median = histogram_median
    self.histogram_variance = histogram_variance
    self.histogram_tendency = histogram_tendency



  def get_data_as_data_frame(self):
    try:
      custom_data_input_dict = {
        "baseline_value": [self.baseline_value],
        "accelerations": [self.accelerations],
        "fetal_movement": [self.fetal_movement],
        "uterine_contractions": [self.uterine_contractions],
        "light_decelerations": [self.light_decelerations],
        "severe_decelerations": [self.severe_decelerations],
        "prolongued_decelerations": [self.prolongued_decelerations],
        "abnormal_short_term_variability": [self.abnormal_short_term_variability],
        "mean_value_of_short_term_variability": [self.mean_value_of_short_term_variability],
        "percentage_of_time_with_abnormal_long_term_variability": [self.percentage_of_time_with_abnormal_long_term_variability],
        "mean_value_of_long_term_variability": [self.mean_value_of_long_term_variability],
        "histogram_width": [self.histogram_width],
        "histogram_min": [self.histogram_min],
        "histogram_max": [self.histogram_max],
        "histogram_number_of_peaks": [self.histogram_number_of_peaks],
        "histogram_number_of_zeroes": [self.histogram_number_of_zeroes],
        "histogram_mode": [self.histogram_mode],
        "histogram_mean": [self.histogram_mean],
        "histogram_median": [self.histogram_median],
        "histogram_variance": [self.histogram_variance],
        "histogram_tendency": [self.histogram_tendency]
      }

      return pd.DataFrame(custom_data_input_dict)
      
    except Exception as e:
      raise CustomException(e, sys)

    