# file for code which is being reused in multiple places
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import ewtpy

import warnings
warnings.filterwarnings('ignore')

###### Initial Data Pull and Data Cleaning ######

def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values , columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime -index
    return df

def raw_to_combined_DF(generation, wind):
    gen_df = get_df(generation)
    wind_df = get_df(wind)
    gen_df = gen_df.resample('3H').mean()
    combined_df = pd.concat([gen_df, wind_df], axis=1)
    combined_df = combined_df.dropna()
    return combined_df

def pull_data(days=90):
    client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread', database='orkney')
    generation = client.query(f'SELECT * FROM Generation WHERE time > now() - {days}d;')
    wind = client.query(f"SELECT * FROM MetForecasts WHERE time > now() - {days}d and time <= now() and Lead_hours = '1';")
    combined_df = raw_to_combined_DF(generation, wind)
    return combined_df

def data_splitting(data, output_val="Total", n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(data):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = data.iloc[train_index][output_val], data.iloc[test_index][output_val]
    return X_train, y_train, X_test, y_test

def final_data_splitting(data, output_val):
    X_train = data
    y_train = data[output_val]
    return X_train, y_train

def load_forecasts():
    # load forecasts
    client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread', database='orkney')
    forecasts = client.query("SELECT * FROM MetForecasts where time > now()")
    forecasts = get_df(forecasts)
    return forecasts

###### Feature Transformation ######

class WindDirectionMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.direction_map = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["DirDeg"] = X["Direction"].map(self.direction_map)
        return X.drop(columns=["Direction"])

    def fit_transform(self, X, y=None):
        return self.transform(X)

class EmpiricalWaveletTransform(BaseEstimator, TransformerMixin):
    def __init__(self, level=5, log=False):
        self.level = level
        self.log = log

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        elems = X.size
        X = X.reshape(elems)
        ewt = ewtpy.EWT1D(X, N=self.level, log=self.log)
        components = []
        for i in range(len(ewt)):
            components.append(ewt[0][:, i].ravel())
        return np.asarray(components).T

class CompassToCartesianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = WindDirectionMapper().fit_transform(X)
        X = np.asarray(X)

        # Convert compass degrees to radians
        radians = np.deg2rad(X)

        # Calculate Cartesian coordinates
        x = np.cos(radians)
        y = np.sin(radians)

        # Stack the x and y coordinates horizontally
        cartesian = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        return cartesian

class WindToComplexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wind_speed_col='Speed', wind_direction_col='Direction'):
        self.wind_speed_col = wind_speed_col
        self.wind_direction_col = wind_direction_col
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        wind_speed = X[self.wind_speed_col]

        wind_direction = WindDirectionMapper().transform(X)
        wind_direction = wind_direction["DirDeg"]
        wind_direction, wind_speed = wind_direction.values, wind_speed.values

        wind_direction_rad = np.radians(wind_direction)
        wind_complex = wind_speed * np.exp(1j * wind_direction_rad)
        return np.vstack([wind_complex.real, wind_complex.imag]).T

class TimestampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, timestamp_col='timestamp', date_col='date', time_col='time_of_day'):
        self.timestamp_col = timestamp_col
        self.date_col = date_col
        self.time_col = time_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        timestamp_index = X.index
        X[self.date_col] = timestamp_index.date
        X[self.time_col] = timestamp_index.time
        return np.asarray(X[[self.date_col, self.time_col]])

###### Performance Metrics ######

def RMSE(true, pred):
    rmse = np.sqrt(np.mean((true - pred)**2))
    return rmse

def MSE(true, pred):
    mse = np.mean((true - pred)**2)
    return mse


###### Plotting ######

def create_timestamps(predict, X_test, y_test):
    predict_timestamp = pd.date_range(start=X_test.index[0], periods=len(predict), freq='3H')
    plot_df = pd.DataFrame({"predict": predict, "actual": y_test}, index=predict_timestamp)
    return plot_df
