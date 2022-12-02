from google.cloud import bigquery
import pandas as pd
import os

PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")

def get_bq_chunk(table: str,
                 index: int,
                 chunk_size: int,
                 verbose=True) -> pd.DataFrame:
    """
    return a chunk of a big query dataset table
    format the output dataframe according to the provided data types
    """
    if verbose:
        print(f"Source data from big query {table}: {chunk_size if chunk_size is not None else 'all'} rows (from row {index})")

    table = f"{PROJECT}.{DATASET}.{table}"

    client = bigquery.Client()

    if verbose:
        if chunk_size is None:
            print(f"\nQuery {table} whole content...")
        else:
            print(f"\nQuery {table} chunk {index // chunk_size} "
                + f"([{index}-{index + chunk_size - 1}])...")

    rows = client.list_rows(table,
                            start_index=index,
                            max_results=chunk_size)

    # convert to expected data types
    big_query_df = rows.to_dataframe()

    if big_query_df.shape[0] == 0:
        return None  # end of data

    big_query_df.rename(columns = {'Unnamed__0':'Unnamed: 0'}, inplace = True)
    return big_query_df
