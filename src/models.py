from flask_sqlalchemy import SQLAlchemy
from .database import db

class RawData(db.Model):
  __tablename__ = 'raw_data'
  id = db.Column(db.Integer, primary_key=True)
  baseline_value = db.Column(db.Float)
  accelerations = db.Column(db.Float)
  fetal_movement = db.Column(db.Float)
  uterine_contractions = db.Column(db.Float)
  light_decelerations = db.Column(db.Float)
  severe_decelerations = db.Column(db.Float)
  prolongued_decelerations = db.Column(db.Float)
  abnormal_short_term_variability = db.Column(db.Float)
  mean_value_of_short_term_variability = db.Column(db.Float)
  percentage_of_time_with_abnormal_long_term_variability = db.Column(db.Float)
  mean_value_of_long_term_variability = db.Column(db.Float)
  histogram_width = db.Column(db.Float)
  histogram_min = db.Column(db.Float)
  histogram_max = db.Column(db.Float)
  histogram_number_of_peaks = db.Column(db.Float)
  histogram_number_of_zeroes = db.Column(db.Float)
  histogram_mode = db.Column(db.Float)
  histogram_mean = db.Column(db.Float)
  histogram_median = db.Column(db.Float)
  histogram_variance = db.Column(db.Float)
  histogram_tendency = db.Column(db.Float)
  fetal_health = db.Column(db.Float)
