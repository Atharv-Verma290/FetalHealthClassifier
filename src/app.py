from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from os import environ
import sys
from src.logger import logging
from src.exception import CustomException
from .database import init_db
from .components.data_ingestion import DataIngestion
from .components.data_transformation import DataTransformation

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DB_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
init_db(app)

data_ingestor = DataIngestion(app=app)
data_ingestor.initiate_data_ingestion()

data_transformer = DataTransformation(app=app)
data_transformer.initiate_data_transformation()


@app.route('/')
def index():
  return "Hello, World! The testint commit"