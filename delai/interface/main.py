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
                                  VALIDATION_DATASET_SIZE,
                                  COLUMN_NAMES_PROCESSED)
from delai.ml_logic.preprocessing import split_X_y, preprocess_X, preprocess_y

from delai.ml_logic.data import get_chunk, save_chunk

from delai.ml_logic.utils import return_dummies_df


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

    from delai.ml_logic.model import (initialize_model, compile_model, train_model)
    from delai.ml_logic.registry import save_model, load_model

    model = None
    model = load_model()  # production model

    # model params
    learning_rate = 0.001
    batch_size = 256
    patience = 2

    # iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    # Grab chunk of processed data (without encoded data)
    while (True):
        print('starting loop')
        data_processed_chunk = get_chunk(source_name=f"train_subset_processed_{DATASET_SIZE}",
                                         index=chunk_id * CHUNK_SIZE,
                                         chunk_size=CHUNK_SIZE)

        if data_processed_chunk is None:
            print(Fore.BLUE + "\nNo more chunk data..." + Style.RESET_ALL)
            break

        # Encode the chunk with all the columns (pd.concat with dummy df)
        encoded_df = return_dummies_df(data_processed_chunk)
        df_tot = pd.DataFrame(columns = COLUMN_NAMES_PROCESSED)
        df_concat = pd.concat([df_tot,encoded_df]).fillna(0)

        print(df_concat.head())

        # Split X and y
        # Create X and y as numpy arrays
        df_X = df_concat.drop(columns=['y','Origin','Dest','Marketing_Airline_Network'])
        y_train = df_concat['y']

        print(df_X.columns)

        X_train = df_X.to_numpy()

        # increment trained row count
        chunk_row_count = data_processed_chunk.shape[0]
        row_count += chunk_row_count

        print(y_train)
        print(X_train)

        # initialize model
        if model is None:
            model = initialize_model(X_train)

        # (re)compile and train the model incrementally
        model = compile_model(model,) #learning_rate)
        model, history = train_model(model,
                                        X_train,
                                        y_train,
                                        batch_size=batch_size,
                                        patience=patience,
                                        # validation_data=(X_val_processed, y_val)
                                        )

        # metrics_val_chunk = np.min(history.history['val_mae'])
        # metrics_val_list.append(metrics_val_chunk)
        # print(f"chunk MAE: {round(metrics_val_chunk,2)}")

        # check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return


# # return the last value of the validation MAE
# val_mae = metrics_val_list[-1]

    print(f"\n✅ trained on {row_count} rows")# with MAE: {round(val_mae, 2)}")

    params = dict(
        # model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        # package behavior
        context="train",
        chunk_size=CHUNK_SIZE,
        # data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=row_count,
        # model_version=get_model_version(),
        # dataset_timestamp=get_dataset_timestamp(),
    )

    # save model
    save_model(model=model, params=params,) #metrics=dict(mae=val_mae))

    print('Completed model training process!')


# return val_mae

#     # From model.py import initialize compile train

#     # Loop back, grab
#     # Save model to the cloud (including params)
#     pass



if __name__ == '__main__':
    #test preprocess function
    #preprocess()
    train()
