import pandas as pd
from delai.ml_logic.params import DATASET_SIZE


def return_dummies_df(df):
    cat_data = df[['Origin','Dest','Marketing_Airline_Network']]
    dest_dummies = pd.get_dummies(cat_data, prefix=['o','d','a'], ).astype(int)
    return df.merge(dest_dummies, left_index=True, right_index=True)

def get_dataset_timestamp(df=None):
    """
    Retrieve the date of the latest available datapoint, at monthly granularity
    """

    import pandas as pd
    from delai.ml_logic.data import get_chunk

    if df is None:
        # Trick specific to this taxifare challenge:
        # Query simply one row from the TRAIN_DATASET, it's enough to deduce the latest datapoint available
        df = get_chunk(source_name=f"train_subset_{DATASET_SIZE}",
                       index=0,
                       chunk_size=1,
                       verbose=False)

    # retrieve first row timestamp
    ts = pd.to_datetime(df.pickup_datetime[:1])[0]

    if ts.year < 2015:
        # Trick specific to this taxifare challenge:
        # We can consider all past training dataset to stop at 2014-12.
        # New datapoints will start to be collected month by month starting 2015-01
        ts = ts.replace(year=2014, month=12)

    # adjust date to monthly granularity
    ts = ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0)

    return ts
