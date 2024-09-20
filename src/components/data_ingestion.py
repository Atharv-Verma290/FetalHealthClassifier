from dataclasses import dataclass
from src.models import RawData, FeatureData, TargetData
from src.database import db
from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
  train_data_path: str=os.path.join('data', 'train.csv')
  test_data_path: str=os.path.join('data', 'test.csv')
  raw_data_path: str=os.path.join('data', 'fetal_health.csv')


class DataIngestion:
  def __init__(self, app):
    self.app = app
    self.ingestion_config = DataIngestionConfig()
    self.db = db

  def initiate_data_ingestion(self):
    logging.info("Entered the data ingestion method or component")
    try:
      logging.info("Starting data ingestion...")
      df = pd.read_csv(self.ingestion_config.raw_data_path)

      #putting raw data into database
      logging.info("Before creating database tables...")
      with self.app.app_context():
        logging.info("Flask app initialized: {}".format(self.app))
      
        logging.info("Creating database tables...")
        db.create_all()
        logging.info("Database tables created successfully")

      # Inserting data into raw_data table
        df.to_sql('raw_data', db.engine, if_exists='replace', index=False)
        db.session.commit()

      os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
      os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

      train_set,test_set, = train_test_split(df,test_size=0.2,random_state=42)

      train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
      logging.info("Data ingestion completed")
    except Exception as e:
      raise CustomException(e,sys)
    
