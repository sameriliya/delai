# importing data from local csvs
import pandas as pd
import os


def get_pandas_chunk(file_name: str):

    path = os.path.join(
        os.path.expanduser(os.environ.get("LOCAL_DATA_PATH")),
        f"{file_name}.csv")

    df = pd.read_csv(
                path)

    #print(df.head())
    return df

def save_pandas_chunk(file_name, data):

    path = os.path.join(
        os.path.expanduser(os.environ.get("LOCAL_DATA_PATH")),
        f"{file_name}.csv")

    data.to_csv(path)
