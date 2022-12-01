# Importing the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_union

def preprocess_X(df_X):
# Dropping columns
    try:
        df_X = df_X.drop(columns = ['Unnamed: 0','Airline','Operating_Airline', 'Flight_Number_Marketing_Airline',
              'OriginStateName', 'OriginCityName','DestStateName', 'DestCityName', 'DestAirportID', 'OriginAirportID', 'FlightDate'])
        df_X = df_X.drop_duplicates()

    except:
        print("Columns don't match input data. Not sure what to clean!")

# Scalling distances

    dist_min = df_X['Distance'].min()
    dist_max = df_X['Distance'].max()

    distance_pipe = make_pipeline(FunctionTransformer(lambda dist: (dist - dist_min)/(dist_max - dist_min)))


# Formatting and Scalling time

    # df_X['FlightDate'] = pd.to_datetime(df_X["FlightDate"])
    # df_X['FlightDate'] = df_X['FlightDate'].dt.dayofweek + 1

    dow = df_X['DayOfWeek']
    sin_dow = np.sin(2 * math.pi / 7 * dow)
    cos_dow = np.cos(2 * math.pi / 7 * dow)

    dom = df_X['DayofMonth']
    sin_dom = np.sin(2 * math.pi / 31 * dom)
    cos_dom = np.cos(2 * math.pi / 31 * dom)

    month = df_X['Month']
    sin_month = np.sin(2 * math.pi / 12 * month)
    cos_month = np.cos(2 * math.pi / 12 * month)

    qua = df_X['Quarter']
    sin_qua = np.sin(2 * math.pi / 4 * qua)
    cos_qua = np.cos(2 * math.pi / 4 * qua)

    dep = df_X['CRSDepTime']
    sin_dep = np.sin(2 * math.pi / 2400 * dep)
    cos_dep = np.cos(2 * math.pi / 2400 * dep)

    arr = df_X['CRSArrTime']
    sin_arr = np.sin(2 * math.pi / 2400 * arr)
    cos_arr = np.cos(2 * math.pi / 2400 * arr)

    #return np.stack([sin_dow,cos_dow, sin_dom, cos_dom, sin_month, cos_month, sin_qua, cos_qua])
    result = pd.DataFrame([sin_dow,cos_dow,sin_dom, cos_dom, sin_month, cos_month, sin_qua, cos_qua, sin_dep,
                      cos_dep, sin_arr, cos_arr]).T
    result.columns = ['sin_dow','cos_dow','sin_dom', 'cos_dom', 'sin_month', 'cos_month', 'sin_qua', 'cos_qua', 'sin_dep',
                      'cos_dep', 'sin_arr', 'cos_arr']
    df_time = pd.DataFrame(result, columns=result.columns)

    df_X = df_X.drop(columns = ['DayOfWeek', 'DayofMonth', 'Month', 'Quarter', 'CRSDepTime', 'CRSArrTime'])

# Creating a joined df

    df_X = pd.merge(df_X, df_time, left_index=True, right_index=True, how = 'outer')

# Encoding categorical values

    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
# print(df_X['Marketing_Airline_Network', 'Origin', 'Dest', 'Year'])
    cat_pipeline = make_column_transformer(
    (cat_transformer, ['Marketing_Airline_Network', 'Origin', 'Dest', 'Year']),
    (distance_pipe, ['Distance']), remainder='passthrough')
    #preprocessor = ColumnTransformer([("dist_preproc", distance_pipe, ['Distance']),],)


# Creating the full pipeline
    #preproc_full = make_union(preprocessor, cat_pipeline)
    X_processed = pd.DataFrame(cat_pipeline.fit_transform(df_X))
    #X_processed = X_processed.drop(columns = [680])
    print("✅ preprocess_X() done")
    return X_processed



def preprocess_y(y, is_binary=True):

    y["DelayGroup"] = None

    if is_binary:
        y.loc[y["ArrDelayMinutes"] == 0, "DelayGroup"] = "InsigDelay"
        y.loc[(y["ArrDelayMinutes"] > 0) & (y["ArrDelayMinutes"] <= 30), "DelayGroup"] = "InsigDelay"
        y.loc[y["ArrDelayMinutes"] > 30, "DelayGroup"] = "Delayed"
        y.loc[y["Cancelled"], "DelayGroup"] = "Delayed"
        y.loc[y["Diverted"], "DelayGroup"] = "Delayed"

    if not is_binary:
        y.loc[y["ArrDelayMinutes"] == 0, "DelayGroup"] = "OnTime_Early"
        y.loc[(y["ArrDelayMinutes"] > 0) & (y["ArrDelayMinutes"] <= 30), "DelayGroup"] = "Small_Delay"
        y.loc[y["ArrDelayMinutes"] > 30, "DelayGroup"] = "Large_Delay"

        y.loc[y["Cancelled"], "DelayGroup"] = "NoArrival"
        y.loc[y["Diverted"], "DelayGroup"] = "NoArrival"

    y = y.drop(columns = ['ArrDelayMinutes', 'Cancelled', 'Diverted', 'Unnamed: 0'])
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(y['DelayGroup'])

    if is_binary:
        print("✅ BINARY preprocess_y() done")
    if not is_binary:
        print("✅ STANDARD preprocess_y() done")
    return encoded_target
