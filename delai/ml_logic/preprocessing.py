# Importing the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder

def split_X_y(df):
    df_X = df.drop(columns = ['ArrDelayMinutes','Cancelled','Diverted'])
    df_y = df[['ArrDelayMinutes','Cancelled','Diverted']]
    return df_X, df_y

def preprocess_X(df_X):
# Encoding year
    df_year = df_X['Year']
    year_encoded = pd.get_dummies(df_year, prefix='y')
    df_X = df_X.merge(year_encoded, left_index=True, right_index=True)
    df_X.drop(columns = 'Year', inplace = True)

# Scaling distances

    dist_min = 16 # grabbed from BQ
    dist_max = 5812 # grabbed from BQ

    df_X['dist_scaled'] = (df_X['Distance'] - dist_min) / (dist_max - dist_min)
    df_X.drop(columns = 'Distance', inplace = True)

# Formatting and Scaling time

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

    df_cols = pd.DataFrame(columns = ['y_2018', 'y_2019','y_2020', 'y_2021', 'y_2022', 'dist_scaled', 'sin_dow', 'cos_dow',
       'sin_dom', 'cos_dom', 'sin_month', 'cos_month', 'sin_qua', 'cos_qua',
       'sin_dep', 'cos_dep', 'sin_arr', 'cos_arr','Marketing_Airline_Network', 'Origin', 'Dest'])
    output = pd.concat([df_cols,df_X]).fillna(0)

    print("✅ preprocess_X() done")
    return output

def preprocess_y(y, is_binary=True):
    y = y.copy()
    y["DelayGroup"] = None

    if is_binary:
        y.loc[y["ArrDelayMinutes"] == 0, "DelayGroup"] = 0
        y.loc[(y["ArrDelayMinutes"] > 0) & (y["ArrDelayMinutes"] <= 30), "DelayGroup"] = 0
        y.loc[y["ArrDelayMinutes"] > 30, "DelayGroup"] = 1
        y.loc[y["Cancelled"], "DelayGroup"] = 1
        y.loc[y["Diverted"], "DelayGroup"] = 1
        # Any remaining None values, assume delayed so turn to 1
        y["DelayGroup"].fillna(1, inplace = True)
        output = y[['DelayGroup']].rename(columns = {'DelayGroup':'y'})


    if not is_binary:
        y.loc[y["ArrDelayMinutes"] == 0, "DelayGroup"] = "OnTime_Early"
        y.loc[(y["ArrDelayMinutes"] > 0) & (y["ArrDelayMinutes"] <= 30), "DelayGroup"] = "Small_Delay"
        y.loc[y["ArrDelayMinutes"] > 30, "DelayGroup"] = "Large_Delay"

        y.loc[y["Cancelled"], "DelayGroup"] = "NoArrival"
        y.loc[y["Diverted"], "DelayGroup"] = "NoArrival"

        y_array = y['DelayGroup']
        label_encoder = LabelEncoder()
        encoded_target = label_encoder.fit_transform(y_array)

        output = pd.DataFrame(encoded_target, columns=['y'])

    if is_binary:
        print("✅ BINARY preprocess_y() done")
    if not is_binary:
        print("✅ STANDARD preprocess_y() done")
    return output
