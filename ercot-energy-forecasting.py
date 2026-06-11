"""
ERCOT Electricity Load & Price Forecasting Proof of Concept (POC)
Author: Milind Verma
Description: Performs univariate/multivariate forecasting on Day-Ahead (DALMP) 
             and Real-Time (RTLMP) Locational Marginal Prices using statistical,
             machine learning, and deep learning models.
"""

import re
import warnings
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.arima.model import ARIMA

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.callbacks import ModelCheckpoint

# Suppress warnings for clean execution
warnings.filterwarnings('ignore')

# Constant for numerical stability in division
EPSILON = 1e-10

# =====================================================================
# 1. CUSTOM EVALUATION METRICS
# =====================================================================

def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    return _error(actual, predicted) / (actual + EPSILON)

def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    return actual[:-seasonality]

def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    if benchmark is None or isinstance(benchmark, int):
        seasonality = 1 if not isinstance(benchmark, int) else benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) / (
            _error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON
        )
    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)

def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    if benchmark is None or isinstance(benchmark, int):
        seasonality = 1 if not isinstance(benchmark, int) else benchmark
        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))
    return abs_err / (abs_err + abs_err_bench + EPSILON)

def _geometric_mean(a, axis=0, dtype=None):
    if not isinstance(a, np.ndarray):
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))

def mse(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(mse(actual, predicted))

def nrmse(actual: np.ndarray, predicted: np.ndarray):
    return rmse(actual, predicted) / (actual.max() - actual.min() + EPSILON)

def me(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(_error(actual, predicted))

def mae(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_error(actual, predicted)))

def gmae(actual: np.ndarray, predicted: np.ndarray):
    return _geometric_mean(np.abs(_error(actual, predicted)))

def mdae(actual: np.ndarray, predicted: np.ndarray):
    return np.median(np.abs(_error(actual, predicted)))

def mpe(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(_percentage_error(actual, predicted))

def mape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_percentage_error(actual, predicted)))

def mdape(actual: np.ndarray, predicted: np.ndarray):
    return np.median(np.abs(_percentage_error(actual, predicted)))

def smape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))

def smdape(actual: np.ndarray, predicted: np.ndarray):
    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))

def maape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))

def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))

def std_ae(actual: np.ndarray, predicted: np.ndarray):
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1))

def std_ape(actual: np.ndarray, predicted: np.ndarray):
    __mape = mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape)) / (len(actual) - 1))

def rmspe(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))

def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))

def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    q = np.abs(_error(actual, predicted)) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))

def inrse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))

def rrse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))

def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    return np.mean(_relative_error(actual, predicted, benchmark))

def rae(actual: np.ndarray, predicted: np.ndarray):
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)

def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))

def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))

def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))

def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))

def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae + EPSILON)

def mda(actual: np.ndarray, predicted: np.ndarray):
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

def bias(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(actual - predicted)

METRICS = {
    "mse": mse, "rmse": rmse, "nrmse": nrmse, "me": me, "mae": mae, "mad": mae,
    "gmae": gmae, "mdae": mdae, "mpe": mpe, "mape": mape, "mdape": mdape,
    "smape": smape, "smdape": smdape, "maape": maape, "mase": mase,
    "std_ae": std_ae, "std_ape": std_ape, "rmspe": rmspe, "rmdspe": rmdspe,
    "rmsse": rmsse, "inrse": inrse, "rrse": rrse, "mre": mre, "rae": rae,
    "mrae": mrae, "mdrae": mdrae, "gmrae": gmrae, "mbrae": mbrae,
    "umbrae": umbrae, "mda": mda, "bias": bias, "r2": r2_score
}

def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=("mae", "rmse", "mape", "r2")):
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print(f"Unable to compute metric {name}: {err}")
    return results

def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))

# =====================================================================
# 2. DATA PROCESSING & FEATURE ENGINEERING
# =====================================================================

def load_and_preprocess_data(filepath="data/POC Sample Data 1.xlsx"):
    print("Loading data...")
    datafrm = pd.read_excel(filepath)
    
    # Preprocess Datetime
    datafrm['DATETIME'] = pd.to_datetime(datafrm['DATETIME'], infer_datetime_format=True)
    df = datafrm.set_index(['DATETIME'])
    
    # Drop irrelevant/redundant columns
    cols_to_drop = ['PEAKTYPE', 'HOURENDING', 'MARKETDAY', 'MONTH', 'YEAR']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(existing_cols_to_drop, axis=1)
    
    # Handle Missing Values via Mean Imputation
    for i in df.columns:
        if df[i].isnull().sum() > 0:
            df[i] = df[i].fillna(df[i].mean())
            
    print("Data processed successfully.")
    return df

def create_time_features(df, target=None):
    df_feat = df.copy()
    df_feat['DATETIME'] = df_feat.index
    df_feat['hour'] = df_feat['DATETIME'].dt.hour
    df_feat['dayofweek'] = df_feat['DATETIME'].dt.dayofweek
    df_feat['quarter'] = df_feat['DATETIME'].dt.quarter
    df_feat['month'] = df_feat['DATETIME'].dt.month
    df_feat['year'] = df_feat['DATETIME'].dt.year
    df_feat['dayofyear'] = df_feat['DATETIME'].dt.dayofyear
    
    # Cyclical representations
    df_feat['sin_day'] = np.sin(df_feat['dayofyear'])
    df_feat['cos_day'] = np.cos(df_feat['dayofyear'])
    df_feat['dayofmonth'] = df_feat['DATETIME'].dt.day
    df_feat['weekofyear'] = df_feat['DATETIME'].dt.isocalendar().week.astype(int)
    
    X = df_feat.drop(['DATETIME'], axis=1)
    if target:
        y = df_feat[target]
        X = X.drop([target], axis=1)
        return X, y
    return X

# =====================================================================
# 3. CORE EXECUTION WRAPPER
# =====================================================================
if __name__ == "__main__":
    # To run this script locally, place "POC Sample Data 1.xlsx" in a folder named "data"
    try:
        df = load_and_preprocess_data()
    except FileNotFoundError:
        print("Please place the 'POC Sample Data 1.xlsx' file inside the 'data/' folder to execute the pipeline.")
