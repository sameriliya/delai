# this is the main interface
# i.e to run preprocessing, training, evaluating, predicting
# calling functions from other .py files
from delai.data.local_disk import get_pandas_chunk
from delai.ml_logic.preprocessing import preprocess_X, preprocess_y
import pandas as pd

if __name__ == '__main__':
    df = get_pandas_chunk('initial_X_train')
    print(df.head())
    output = preprocess_X(df)
    print(output.head())

    df_y = get_pandas_chunk('initial_y_train')
    y_output = preprocess_y(df_y)
    print(y_output)
