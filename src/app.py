from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from os import environ
import sys
from src.logger import logging
from src.exception import CustomException
from .database import init_db
from .components.data_ingestion import DataIngestion
from .components.data_transformation import DataTransformation
from .pipeline.train_pipeline import TrainPipeline

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DB_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
init_db(app)

logging.info("Calling train pipeline")
train_pipeline = TrainPipeline()
train_pipeline.initiate_training()

@app.route('/')
def index():
  return "Hello, World! The testint commit"