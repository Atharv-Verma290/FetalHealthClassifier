import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.joblib")
  label_encoder_obj_file_path = os.path.join('artifacts',"label_encoder.joblib")

class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()


  def get_data_transformer_object(self):
    try:
      numerical_columns = ["baseline_value", "accelerations", "uterine_contractions", "light_decelerations", "severe_decelerations", "prolongued_decelerations", "abnormal_short_term_variability",
                           "mean_value_of_short_term_variability", "percentage_of_time_with_abnormal_long_term_variability", "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
                           "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes", "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance", "histogram_tendency"]
      
      categorical_columns = ["fetal_health"]

      num_pipeline = Pipeline(
        steps=[
          ('scaler', MinMaxScaler())
        ]
      )

      logging.info(f"Categorical column: {categorical_columns}")
      logging.info(f"Numerical columns: {numerical_columns}")

      preprocessor = ColumnTransformer(
        [
          ("num_pipeline",num_pipeline,numerical_columns)
        ]
      )

      return preprocessor
    except Exception as e:
      raise CustomException(e, sys)


  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)

      logging.info("read train and test data completed")
      logging.info("Obtaining preprocessing object")

      preprocessor = self.get_data_transformer_object()

      target_column_name = "fetal_health"

      input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
      target_feature_train_df = train_df[target_column_name]

      input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
      target_feature_test_df = test_df[target_column_name]

      mapping = {1.0: 'Normal', 2.0: 'Suspect', 3.0: 'Pathological'}
      target_feature_train_df_mapped = target_feature_train_df.map(mapping)
      target_feature_test_df_mapped = target_feature_test_df.map(mapping)
      logging.info("target_feature mapped in train and test df")

      logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
      # Apply the feature preprocessor to transform the input features       
      input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessor.transform(input_feature_test_df)

      logging.info("Applying label encoding on target feature")
      # Apply LabelEncoder on the target column separately
      label_encoder = LabelEncoder()
      target_feature_train_arr= label_encoder.fit_transform(target_feature_train_df_mapped)
      target_feature_test_arr = label_encoder.transform(target_feature_test_df_mapped)

      train_arr = np.c_[
        input_feature_train_arr, np.array(target_feature_train_arr)
      ]
      test_arr = np.c_[
        input_feature_test_arr, np.array(target_feature_test_arr)
      ]

      logging.info("Saving preprocessing objects.")

      save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessor
      )

      save_object(
        file_path=self.data_transformation_config.label_encoder_obj_file_path,
        obj=label_encoder
      )

      return(
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path,
        self.data_transformation_config.label_encoder_obj_file_path
      )
    
    except Exception as e:
      raise CustomException(e, sys)
