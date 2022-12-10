from delai.data.local_disk import (get_pandas_chunk, save_pandas_chunk)
from delai.data.big_query import (get_bq_chunk, save_bq_chunk)
from delai.ml_logic.params import (COLUMN_NAMES_RAW, COLUMN_NAMES_PART_PROCESSED)
import os
import pandas as pd

def get_chunk(source_name: str,
              index: int = 0,
              chunk_size: int = None,
              verbose=False) -> pd.DataFrame:
    """
    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
    Always assumes `source_name` (CSV or Big Query table) have headers,
    and do not consider them as part of the data `index` count.
    """
    if "processed" in source_name:
        columns = COLUMN_NAMES_PART_PROCESSED
    else:
        columns = COLUMN_NAMES_RAW

    if os.environ.get("DATA_SOURCE") == "big query":

        chunk_df = get_bq_chunk(table=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                verbose=verbose)

        return chunk_df

    chunk_df = get_pandas_chunk(path=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                columns=columns,
                                verbose=verbose)

    return chunk_df

def save_chunk(destination_name: str,
               is_first: bool,
               data: pd.DataFrame) -> None:
    """
    save chunk
    """

    if os.environ.get("DATA_SOURCE") == "big query":

        save_bq_chunk(table=destination_name,
                      data=data,
                      is_first=is_first)

        return

    save_pandas_chunk(path=destination_name,
                     data=data,
                     is_first=is_first)
