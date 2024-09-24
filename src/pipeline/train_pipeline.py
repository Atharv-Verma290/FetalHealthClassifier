import os
import sys
from src.logger import logging
from src.exception import CustomException
from ..components.data_ingestion import DataIngestion
from ..components.data_transformation import DataTransformation
from ..components.model_trainer import ModelTrainer


class TrainPipeline:
  def __init__(self):
    pass

  def initiate_training(self):
    try:
      obj = DataIngestion()
      train_data, test_data = obj.initiate_data_ingestion()

      data_transformation = DataTransformation()
      train_arr, test_arr,_,_ = data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)

      model_trainer = ModelTrainer()
      logging.info(f"the r2 score of best model: {model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)}")
      # print(model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))
    except Exception as e:
      raise CustomException(e, sys)

  