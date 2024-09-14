from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from os import environ
from src.db_init import db
from src.models import RawData
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DB_URL')
db.init_app(app)

csv_file_path = "data/fetal_health.csv"
df = pd.read_csv(csv_file_path)

with app.app_context():
  db.create_all()
  df.to_sql('raw_data', db.engine, if_exists='replace', index=False)

@app.route('/')
def index():
  return "Hello, World!"