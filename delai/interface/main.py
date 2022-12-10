import numpy as np
import pandas as pd
import datetime

from delai.ml_logic.params import (CHUNK_SIZE,
                                   DATASET_SIZE,
                                  VALIDATION_DATASET_SIZE,
                                  COLUMN_NAMES_PROCESSED)
from delai.ml_logic.preprocessing import split_X_y, preprocess_X, preprocess_y
from delai.ml_logic.data import get_chunk, save_chunk
from delai.ml_logic.utils import return_dummies_df, get_dataset_timestamp
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
        # preprocess the X and Y DataFrames into desired output
        X_processed_chunk = preprocess_X(df_X)
        y_processed_chunk = preprocess_y(df_y)
        data_processed_chunk = pd.merge(X_processed_chunk, y_processed_chunk,
                                        left_index=True, right_index=True)
        # helpful stats for terminal
        print(data_processed_chunk.loc[data_processed_chunk.isna().any(axis=1)])
        print('X_proc_shape:', X_processed_chunk.shape)
        print('y_proc_shape:', y_processed_chunk.shape)
        print(data_processed_chunk.head())
        print('full_data_shape:', data_processed_chunk.shape)
        print('full_data_shape_noNA:', data_processed_chunk.dropna().shape)

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
    """Train model chunk by chunk. Either to MLFlow or Locally.
    Will check for existing model in production on MLFlow and try to update weights
    if calling on training again"""
    print("\n⭐️ use case: train")
    from delai.ml_logic.model import (initialize_model, compile_model, train_model)
    from tensorflow import convert_to_tensor, float32, int64
    from delai.ml_logic.registry import save_model, load_model, get_model_version
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # load a validation set common to all chunks, used to early stop model training
    data_val_processed = get_chunk(
        source_name=f"val_subset_processed_{VALIDATION_DATASET_SIZE}",
        index=0,  # retrieve from first row
        chunk_size=None)  # retrieve all further data

    if data_val_processed is None:
        print("\n✅ no data to train")
        return None

    #Do our post processing to encode Origin / Dest / Airline
    encoded_df = return_dummies_df(data_val_processed)
    df_tot = pd.DataFrame(columns = COLUMN_NAMES_PROCESSED)
    df_concat = pd.concat([df_tot,encoded_df]).fillna(0)
    # Split X and y
    # Create X and y as numpy arrays
    df_X = df_concat.drop(columns=['y','Origin','Dest','Marketing_Airline_Network'])
    y_val = df_concat['y']
    y_val = convert_to_tensor(y_val, dtype=int64)
    X_val_processed = df_X.to_numpy()
    X_val_processed = convert_to_tensor(X_val_processed, dtype=float32)

    model = None
    model = load_model()  # production model

    # model params
    learning_rate = 0.001
    batch_size = 32
    patience = 10

    # iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    # Grab chunk of processed data (without encoded data)
    while (True):
        print(f'Starting to train chunk {chunk_id}')
        data_processed_chunk = get_chunk(source_name=f"train_subset_processed_{DATASET_SIZE}",
                                         index=chunk_id * CHUNK_SIZE,
                                         chunk_size=CHUNK_SIZE)

        if data_processed_chunk is None:
            print(Fore.BLUE + "\nNo more chunk data..." + Style.RESET_ALL)
            break

        # Encode the chunk with all the columns (pd.concat with dummy df)
        print(data_processed_chunk.head())
        encoded_df = return_dummies_df(data_processed_chunk)
        df_tot = pd.DataFrame(columns = COLUMN_NAMES_PROCESSED)
        df_concat = pd.concat([df_tot,encoded_df]).fillna(0)

        print(df_concat.head())

        # Split X and y
        # Create X and y as numpy arrays
        df_X = df_concat.drop(columns=['y','Origin','Dest','Marketing_Airline_Network'])
        y_train = df_concat['y']
        y_train = convert_to_tensor(y_train, dtype=int64)

        print(df_X.columns)

        X_train = df_X.to_numpy()
        X_train = convert_to_tensor(X_train, dtype=float32)

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
                                        validation_data=(X_val_processed, y_val)
                                        )

        metrics_val_chunk = np.min(history.history['val_accuracy'])
        metrics_val_list.append(metrics_val_chunk)
        print(f"chunk accuracy: {round(metrics_val_chunk,2)}")

        # check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return


    # # return the last value of the validation accuracy

    val_accuracy = metrics_val_list[-1]
    # Remember we'll need to change this on new metrics from Michael's

    print(f"\n✅ trained on {row_count} rows with accuracy: {round(val_accuracy, 2)}")

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
        model_version=get_model_version()
    )

    # save model
    save_model(model=model, params=params, metrics=dict(accuracy=val_accuracy))

    print('Completed model training process!')

    return val_accuracy

def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """
    print("\n⭐️ use case: evaluate")

    from delai.ml_logic.model import evaluate_model
    from delai.ml_logic.registry import load_model, save_model
    from delai.ml_logic.registry import get_model_version
    from tensorflow import convert_to_tensor, float32, int64
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


    # load new data
    new_data = get_chunk(source_name=f"val_subset_processed_{DATASET_SIZE}",
                         index=0,
                         chunk_size=None)  # retrieve all further data

    if new_data is None:
        print("\n✅ no data to evaluate")
        return None

    #Do our post processing to encode Origin / Dest / Airline
    encoded_df = return_dummies_df(new_data)
    df_tot = pd.DataFrame(columns = COLUMN_NAMES_PROCESSED)
    df_concat = pd.concat([df_tot,encoded_df]).fillna(0)
    # Split X and y
    # Create X and y as numpy arrays
    df_X = df_concat.drop(columns=['y','Origin','Dest','Marketing_Airline_Network'])
    y_new = df_concat['y']
    y_new = convert_to_tensor(y_new, dtype=int64)
    X_new = df_X.to_numpy()
    X_new = convert_to_tensor(X_new, dtype=float32)
    print(df_X.shape)
    print(df_X.columns)
    #Continue with same logic as taxifare
    model = load_model()
    if model == None:
        print('No model, stopping Evaluate')
        return

    # metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    y_pred = model.predict(X_new)
    pd.DataFrame(y_pred).to_csv('raw_data/y_pred_unrounded.csv', index = False)
    y_pred = np.round(y_pred).astype(int)
    pd.DataFrame(y_pred).to_csv('raw_data/y_pred_rounded.csv', index = False)
    pd.DataFrame(y_new).to_csv('raw_data/y_actual.csv', index = False)

    acc_man = round(accuracy_score(y_new, y_pred), 3)
    prec_man = round(precision_score(y_new, y_pred), 3)
    rec_man = round(recall_score(y_new, y_pred), 3)
    f1_man = round(f1_score(y_new, y_pred), 3)

    # acc = metrics_dict["accuracy"] #using accuracy here as classification

    # save evaluation
    params = dict(
        #dataset_timestamp=get_dataset_timestamp(),
        model_version=get_model_version(),
        # package behavior
        context="evaluate",
        # data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=len(X_new))

    save_model(params=params, metrics=dict(#accuracy=acc,
                                           acc_man=acc_man,
                                           precision=prec_man,
                                           recall = rec_man,
                                           f1 =f1_man))
    print('New metrics:', acc_man, prec_man, rec_man, f1_man)
    return f1_man

def pred(flight_number='DAL383', date=datetime.date.today()) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ use case: predict")

    from delai.ml_logic.registry import load_model
    from delai.api.flightaware import get_processed_flight_details

    # get details of new flight from API call, and remove column to pass into preprocessor
    df_new = get_processed_flight_details(flight_number, date).drop(columns = 'FlightDate')
    X_processed = preprocess_X(df_new)
    print(X_processed)

    encoded_df = return_dummies_df(X_processed)
    df_tot = pd.DataFrame(columns = COLUMN_NAMES_PROCESSED)
    df_concat = pd.concat([df_tot,encoded_df]).fillna(0)
    # Split X and y
    # Create X and y as numpy arrays
    X_new = df_concat.drop(columns=['y','Origin','Dest','Marketing_Airline_Network'])
    print(X_new)
    model = load_model()
    y_pred = model.predict(X_new.to_numpy())

    print("\n✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred

if __name__ == '__main__':
    #test preprocess function
    #preprocess()
    #preprocess(source_type = 'val_subset')
    train()
    # evaluate()
    #pred()
