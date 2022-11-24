# Importing the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import make_union


# Dropping columns

def data_cleaning(X):
    try:
        X = X.drop(columns = ['Unnamed: 0','Airline','Operating_Airline', 'Flight_Number_Marketing_Airline',
              'OriginStateName', 'OriginCityName','DestStateName', 'DestCityName', 'DestAirportID', 'OriginAirportID'])
        X = X.drop_duplicates()

    except:pass

    return X


# Scalling distances

def distance_scalling(X):
    dist_min = X['Distance'].min()
    dist_max = X['Distance'].max()

    distance_pipe = make_pipeline(FunctionTransformer(lambda dist: (dist - dist_min)/(dist_max - dist_min)))

    return distance_pipe


# Formatting and Scalling time

def time_format(X):
    X['FlightDate'] = pd.to_datetime(X["FlightDate"])
    X['FlightDate'].dt.dayofweek
    X['FlightDate'].dt.dayofweek

    return (X['FlightDate'].dt.dayofweek) + 1


def transform_time_features(X: pd.DataFrame):
    dow = X['DayOfWeek']
    sin_dow = np.sin(2 * math.pi / 7 * dow)
    cos_dow = np.cos(2 * math.pi / 7 * dow)

    dom = X['DayofMonth']
    sin_dom = np.sin(2 * math.pi / 31 * dom)
    cos_dom = np.cos(2 * math.pi / 31 * dom)

    month = X['Month']
    sin_month = np.sin(2 * math.pi / 12 * month)
    cos_month = np.cos(2 * math.pi / 12 * month)

    qua = X['Quarter']
    sin_qua = np.sin(2 * math.pi / 4 * qua)
    cos_qua = np.cos(2 * math.pi / 4 * qua)

    dep = X['CRSDepTime']
    sin_dep = np.sin(2 * math.pi / 2400 * qua)
    cos_dep = np.cos(2 * math.pi / 2400 * qua)

    arr = X['CRSArrTime']
    sin_arr = np.sin(2 * math.pi / 2400 * qua)
    cos_arr = np.cos(2 * math.pi / 2400 * qua)

    #return np.stack([sin_dow,cos_dow, sin_dom, cos_dom, sin_month, cos_month, sin_qua, cos_qua])
    result = pd.DataFrame([sin_dow,cos_dow,sin_dom, cos_dom, sin_month, cos_month, sin_qua, cos_qua, sin_dep,
                      cos_dep, sin_arr, cos_arr]).T
    result.columns = ['sin_dow','cos_dow','sin_dom', 'cos_dom', 'sin_month', 'cos_month', 'sin_qua', 'cos_qua', 'sin_dep',
                      'cos_dep', 'sin_arr', 'cos_arr']

    return result


# Creating a joined df

def updated_df(X):
    df_time = transform_time_features(X)
    df_X = pd.merge(df_X, df_time, left_index=True, right_index=True, how = 'outer')
    df_X = df_X.drop(columns = ['Unnamed: 0'])


# Encoding categorical values
