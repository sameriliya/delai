# this is the main interface
# i.e to run preprocessing, training, evaluating, predicting
# calling functions from other .py files
from delai.data.local_disk import get_pandas_chunk
from delai.ml_logic.preprocessing import split_X_y, preprocess_X, preprocess_y
from delai.data.big_query import get_bq_chunk
import pandas as pd

if __name__ == '__main__':
    df = get_pandas_chunk('train_100k')
    print(df.head().dtypes)
    df_X, df_y = split_X_y(df)


    X_output = preprocess_X(df_X)
    print(X_output.head())

    y_output = preprocess_y(df_y)
    print(y_output)

    # testing retreving data from big_query and preprocessing
    bq_df = get_bq_chunk(table = 'train_100k', index = 0, chunk_size = 1000)
    print(bq_df.dtypes)
    bq_df_X, bq_df_y = split_X_y(bq_df)

    bqX_output = preprocess_X(bq_df_X)
    print(bqX_output.head())

    bqy_output = preprocess_y(bq_df_y)
    print(bqy_output)
