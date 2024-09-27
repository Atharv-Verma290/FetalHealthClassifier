import os 
import sys
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
      joblib.dump(obj, file_obj)
  
  except Exception as e:
    raise CustomException(e, sys)
  
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
  try:
    report = {}

    for i in range(len(list(models))):
      model = list(models.values())[i]
      para = param[list(models.keys())[i]]

      gs = RandomizedSearchCV(model,para,cv=3)
      gs.fit(X_train, y_train)

      model.set_params(**gs.best_params_)
      model.fit(X_train, y_train)

      # model.fit(X_train, y_train) #Train model

      y_train_pred = model.predict(X_train)
      y_test_pred = model.predict(X_test)

      train_model_score = r2_score(y_train, y_train_pred)
      test_model_score = r2_score(y_test, y_test_pred)

      report[list(models.keys())[i]] = test_model_score

    return report
  
  except Exception as e:
    raise CustomException(e, sys)
  

def load_object(file_path):
  try:
    with open(file_path, "rb") as file_obj:
      return joblib.load(file_obj)
    
  except Exception as e:
    raise CustomException(e, sys)
  

def process_result(result):
  try:
    label_encoder_path = 'artifacts/label_encoder.joblib'
    label_encoder = load_object(label_encoder_path)
    result = [result]
    result = [int(r) for r in result]
    output = label_encoder.inverse_transform(result)
    return output
  except Exception as e:
    raise CustomException(e, sys)