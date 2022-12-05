# this is the main interface
# i.e to run preprocessing, training, evaluating, predicting
# calling functions from other .py files
from delai.data.local_disk import get_pandas_chunk
from delai.ml_logic.model import test_model_run
from delai.data.big_query import get_bq_chunk
from delai.api.flightaware import get_processed_flight_details
from delai.ml_logic.registry import save_model

import numpy as np
import pandas as pd

from delai.ml_logic.params import (CHUNK_SIZE,
                                   DATASET_SIZE,
                                  VALIDATION_DATASET_SIZE)
from delai.ml_logic.preprocessing import split_X_y, preprocess_X, preprocess_y

from delai.ml_logic.data import get_chunk, save_chunk


from colorama import Fore, Style

def preprocess(source_type='train_subset'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    print("\n⭐️ use case: preprocess")

    # iterate on the dataset, by chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{source_type}_{DATASET_SIZE}"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}"

    if DATASET_SIZE == 'full':
        source_name = f"{source_type}"

    while (True):

        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(source_name=source_name,
                               index=chunk_id * CHUNK_SIZE,
                               chunk_size=CHUNK_SIZE)

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]
        df_X, df_y = split_X_y(data_chunk)

        # break out of while loop if cleaning removed all rows
        if len(df_X) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break

        X_processed_chunk = preprocess_X(df_X)
        y_processed_chunk = preprocess_y(df_y)
        data_processed_chunk = pd.DataFrame(
            pd.concat((X_processed_chunk, y_processed_chunk), axis=1))

        # save and append the chunk
        is_first = chunk_id == 0

        save_chunk(destination_name=destination_name,
                   is_first=is_first,
                   data=data_processed_chunk)

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the preprocessing 👌")
        return None

    print(f"\n✅ data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None

def train():
    '''
    M's code on model training chunk by chunk
    '''
    pass



if __name__ == '__main__':
    # df = get_pandas_chunk('train_100k')
    # print(df.head().dtypes)
    # df_X, df_y = split_X_y(df)

    # X_output = preprocess_X(df_X)
    # print(X_output.head())

    # y_output = preprocess_y(df_y)
    # print(y_output)

    # testing retreving data from big_query and preprocessing
    # bq_df = get_bq_chunk(table = 'train_100k', index = 0, chunk_size = 1000)
    # print(bq_df.dtypes)
    # bq_df_X, bq_df_y = split_X_y(bq_df)
    # print(bq_df_X)

    # bqX_output = preprocess_X(bq_df_X)
    # print(bqX_output.head())

    # bqy_output = preprocess_y(bq_df_y)
    # print(bqy_output)

    # model,history = test_model_run(bqX_output,bqy_output)
    # print('Model has been fitted successfully')

    # save_model(model)

    # X_new = get_processed_flight_details()
    # print(bq_df_X.dtypes)
    # print(X_new.dtypes)
    # X_new = preprocess_X(X_new)
    # print('processed sample flight')
    # print(X_new)

    #test preprocess function
    preprocess()
    # df = pd.read_csv('raw_data/raw/train_100k.csv')
    # df = get_bq_chunk(table = 'train', index = 0, chunk_size = 100000)
    # print(df.columns)
    # df_X, df_y = split_X_y(df)
    # X_output = preprocess_X(df_X)
    # print(X_output.head())
    # y_output = preprocess_y(df_y)
    # print(y_output)
    # X_output.to_csv('../raw_data/m_s_train_processed.csv', index = False)
