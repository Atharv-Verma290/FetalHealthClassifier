from flask import Flask, request, jsonify, make_response, render_template
from flask_sqlalchemy import SQLAlchemy
from os import environ
import sys
from src.logger import logging
from src.exception import CustomException
from .database import init_db
from .components.data_ingestion import DataIngestion
from .components.data_transformation import DataTransformation
from .pipeline.train_pipeline import TrainPipeline
from .pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DB_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
init_db(app)

# logging.info("Calling train pipeline")
# train_pipeline = TrainPipeline()
# train_pipeline.initiate_training()

@app.route('/')
def index():
  return "Hello, World! The testint commit"

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
  if request.method == 'GET':
    return render_template('home.html')
  else:
    data = CustomData(
      baseline_value=request.form.get('baseline_value'),
      accelerations=request.form.get('accelerations'),
      fetal_movement=request.form.get('fetal_movement'),
      uterine_contractions=request.form.get('uterine_contractions'),
      light_decelerations=request.form.get('light_decelerations'),
      severe_decelerations=request.form.get('severe_decelerations'),
      prolongued_decelerations=request.form.get('prolongued_decelerations'),
      abnormal_short_term_variability=request.form.get('abnormal_short_term_variability'),
      mean_value_of_short_term_variability=request.form.get('mean_value_of_short_term_variability'),
      percentage_of_time_with_abnormal_long_term_variability=request.form.get('percentage_of_time_with_abnormal_long_term_variability'),
      mean_value_of_long_term_variability=request.form.get('mean_value_of_long_term_variability'),
      histogram_width=request.form.get('histogram_width'),
      histogram_min=request.form.get('histogram_min'),
      histogram_max=request.form.get('histogram_max'),
      histogram_number_of_peaks=request.form.get('histogram_number_of_peaks'),
      histogram_number_of_zeroes=request.form.get('histogram_number_of_zeroes'),
      histogram_mode=request.form.get('histogram_mode'),
      histogram_mean=request.form.get('histogram_mean'),
      histogram_median=request.form.get('histogram_median'),
      histogram_variance=request.form.get('histogram_variance'),
      histogram_tendency=request.form.get('histogram_tendency')
    )
    pred_df = data.get_data_as_data_frame()
    print(pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    print(results)
    return render_template('home.html', results=results)
  

    