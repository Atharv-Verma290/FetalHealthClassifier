import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.database import db
from src.models import RawData, FeatureData, TargetData


@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
  def __init__(self, app):
    self.app = app
    self.db  = db


  def initiate_data_transformation(self):
    logging.info("Entered data transformation component")
    
    try:
      logging.info("Connecting to db")
      with self.app.app_context():

        logging.info("Fetching raw data from raw_data")
        query = "SELECT * FROM raw_data"
        raw_df = pd.read_sql_query(query, db.engine)
        logging.info("Fetched raw data successfully")

      # Transforming Raw Data
      logging.info("Transforming raw data")
      mapping = {1.0: 'Normal', 2.0: 'Suspect', 3.0: 'Pathological'}
      raw_df['fetal_health'] = raw_df['fetal_health'].map(mapping)

      logging.info("Target Column transformed to categorical")

      logging.info("Begin One Hot Encoding on target column")
      df_encoded = pd.get_dummies(raw_df, columns=['fetal_health'])

      
      X = df_encoded.drop(columns=['fetal_health_Normal', 'fetal_health_Suspect', 'fetal_health_Pathological'], axis=1)
      y = df_encoded[['fetal_health_Normal', 'fetal_health_Suspect', 'fetal_health_Pathological']].astype(int)

      logging.info("encoding completed")

      logging.info("adding X and y data in db")
      with self.app.app_context():
        X.to_sql('feature_data', db.engine, if_exists='replace', index=False)
        db.session.commit()
        y.to_sql('target_data', db.engine, if_exists='replace', index=False)
        db.session.commit()

      logging.info("Processed Data added to db successfully")

    except Exception as e:
      raise CustomException(e,sys)