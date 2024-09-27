from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from os import environ
import sys
from src.logger import logging
from src.exception import CustomException
from .database import init_db
from .pipeline.train_pipeline import TrainPipeline
from .pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DB_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
init_db(app)

logging.info("Calling train pipeline")
train_pipeline = TrainPipeline(app=app)
train_pipeline.initiate_training()

@app.route('/')
def index():
  return "Hello, World! The testint commit"
  
@app.route("/predictdata", methods=['POST'])
def predict_datapoint():
  data_json = request.get_json()
  data = CustomData(
      baseline_value=data_json.get('baseline_value'),
      accelerations=data_json.get('accelerations'),
      fetal_movement=data_json.get('fetal_movement'),
      uterine_contractions=data_json.get('uterine_contractions'),
      light_decelerations=data_json.get('light_decelerations'),
      severe_decelerations=data_json.get('severe_decelerations'),
      prolongued_decelerations=data_json.get('prolongued_decelerations'),
      abnormal_short_term_variability=data_json.get('abnormal_short_term_variability'),
      mean_value_of_short_term_variability=data_json.get('mean_value_of_short_term_variability'),
      percentage_of_time_with_abnormal_long_term_variability=data_json.get('percentage_of_time_with_abnormal_long_term_variability'),
      mean_value_of_long_term_variability=data_json.get('mean_value_of_long_term_variability'),
      histogram_width=data_json.get('histogram_width'),
      histogram_min=data_json.get('histogram_min'),
      histogram_max=data_json.get('histogram_max'),
      histogram_number_of_peaks=data_json.get('histogram_number_of_peaks'),
      histogram_number_of_zeroes=data_json.get('histogram_number_of_zeroes'),
      histogram_mode=data_json.get('histogram_mode'),
      histogram_mean=data_json.get('histogram_mean'),
      histogram_median=data_json.get('histogram_median'),
      histogram_variance=data_json.get('histogram_variance'),
      histogram_tendency=data_json.get('histogram_tendency')
    )
  pred_df = data.get_data_as_data_frame()
  # print(pred_df)

  predict_pipeline = PredictPipeline()
  result = predict_pipeline.predict(pred_df)
  result = result.tolist()
  logging.info(f"Output is: {result}")
  return jsonify({'result': result}, 200)