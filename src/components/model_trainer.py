from dataclasses import dataclass
import sys
import os
from src.logger import logging
from src.exception import CustomException
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join("artifacts","model.joblib")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
    logging.info("Entered model trainer component")
    try: 
      logging.info("Splitting X and y data")
      X_train, y_train, X_test, y_test = (
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      )

      models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
      }
      logging.info(f"models list: {models}")
      params={
        "Decision Tree": {
          'criterion': ['gini', 'entropy'],
          'splitter' : ['best', 'random'],
          'max_features': ['sqrt', 'log2'],
          'max_depth': [2, 5, 8, 10, None],
          'min_samples_split': [2, 5, 10],
        },
        "Random Forest": {
          'criterion': ['gini','entropy'],
          'max_features': ['sqrt', 'log2', None],
          'n_estimators': [8, 16, 32, 64, 128, 256],
          'max_features': [1,3,5,7],
          'min_samples_leaf': [1,2,3],
          'min_samples_split': [2,3,6],
          'max_depth': [2,8,None],
          'max_samples': [0.5,0.75,1.0]
        }
      }

      model_report: dict = evaluate_model(X_train=X_train, y_train=y_train,X_test = X_test, y_test = y_test, models = models, param = params)

      # To get best model score from dict
      best_model_score = max(sorted(model_report.values()))

      #To get best model name from dict
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

      best_model = models[best_model_name]

      if best_model_score < 0.6:
        raise CustomException("No best model found")
      
      logging.info(f"Best model found on both training and testing dataset")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      predicted = best_model.predict(X_test)
      r2_square = r2_score(y_test, predicted)
      return r2_square

    except Exception as e:
      raise CustomException(e,sys)