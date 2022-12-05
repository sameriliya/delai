import pandas as pd

def return_dummies_df(df):
    cat_data = df[['Origin','Dest','Marketing_Airline_Network']]
    dest_dummies = pd.get_dummies(cat_data, prefix=['o','d','a'], ).astype(int)
    return df.merge(dest_dummies, left_index=True, right_index=True)
