from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import warnings
import sys
import datetime as dt
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, AzureCliCredential, InteractiveBrowserCredential
import os
import argparse

warnings.filterwarnings('ignore')
sys.path.append('..')
import fx

parser = argparse.ArgumentParser()
parser.add_argument("-tracking_server", help="Tracking server to use", default="local")
parser.add_argument("-days", help="Number of days to pull data from", default=90)
args = parser.parse_args()

tracking_server = args.tracking_server
days = args.days
print(f"Running sklearn pipeline on tracking server: {tracking_server} and {days} days of data")

if tracking_server == "itu-training":
    mlflow.set_tracking_uri("http://training.itu.dk:5000/")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://130.226.140.28:5000"
    os.environ["AWS_ACCESS_KEY_ID"] = "training-bucket-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "tqvdSsEDnBWTDuGkZYVsRKnTeu"

elif tracking_server == "azure":
    ml_client = MLClient.from_config(credential=AzureCliCredential())
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

elif tracking_server == "local":
    # mlflow.set_tracking_uri("http://localhost:5000")
    pass

data = fx.pull_data(days)

pipeline = Pipeline(steps=[
    ("col_transformer", ColumnTransformer(transformers=[
        ("time", None, []),
        ("Speed", None, ["Speed"]),
        ("Direction", None, ["Direction"]),
        ], remainder="drop")),
    ("model", None)
])

params = {
    'col_transformer__time' : ["drop", None, fx.TimestampTransformer()],
    'col_transformer__Speed': [None, StandardScaler(), PolynomialFeatures(), fx.EmpiricalWaveletTransform(level=5)],
    'col_transformer__Direction': ["drop", fx.WindDirectionMapper(), fx.CompassToCartesianTransformer()],
    'model': [
        LinearRegression(), 
        MLPRegressor(hidden_layer_sizes=(150, 150), activation='tanh', solver='sgd'), 
        SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1),
        HuberRegressor(epsilon=1.35, alpha=0.0001),
        RANSACRegressor(min_samples=0.1, max_trials=100),
        GaussianProcessRegressor(alpha=0.1, kernel=RBF()) 
    ]
}

tscv = TimeSeriesSplit(n_splits=5)

scorer = "neg_mean_absolute_percentage_error"

gridsearch = GridSearchCV(pipeline, params, cv=tscv, scoring=scorer, n_jobs=-1, verbose=1)

X_train, y_train, X_test, y_test = fx.data_splitting(data, output_val="Total")

mlflow.start_run()
# mlflow.set_experiment("Orkney-Windpower-Prediction")

mlflow.log_param("days", days)

gridsearch.fit(X_train, y_train)

print("logging model")
mlflow.sklearn.log_model(gridsearch, "Model")

print("predicting")
predictions = gridsearch.predict(X_test)

print("logging metrics")
mlflow.log_metric("test_mse", fx.MSE(y_test, predictions))

print("Done")

