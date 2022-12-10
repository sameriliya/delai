from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from delai.ml_logic.preprocessing import preprocess_X, preprocess_y
from delai.ml_logic.registry import load_model
import pytz

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()


# http://127.0.0.1:8000/predict?flight_date=2022-12-09&flight_number=DAL383
@app.get("/predict")

def predict(date: datetime,  # datetime.date.today()
            flight_number: str):    # DAL383

    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """

    key = str(date)
    eastern = pytz.timezone("US/Eastern")
    localized_flight_date = eastern.localize(date, is_dst=None)
    # convert the user datetime to UTC and format the datetime as expected by the pipeline
    utc_flight_date = localized_flight_date.astimezone(pytz.utc)
    formatted_flight_date = utc_flight_date.strftime("%Y-%m-%d UTC")

    X_pred = pd.DataFrame({"key":[key],
                       "date":[formatted_flight_date],
                       "flight_number":[flight_number]})

    model = app.state.model

    X_preprocessed = preprocess_X(X_pred)

    y_predict = model.predict(X_preprocessed)

    return {'fare_amount':round(float(y_predict),2)}

@app.get("/")
def root():
    response = {'Delai prediction': 'test'}

    return response


if __name__ == "__main__":
    test = predict(flight_date="2022-12-09",
                   flight_number="DAL383")

    # print(test)
    import pytest
    from httpx import AsyncClient
    import os

    test_params = {
        'flight_date': '2022-12-09',
        'flight_number': 'DAL383'
    }

    # @pytest.mark.skipif(TEST_ENV != "development", reason="only dev mode")
    # @pytest.mark.asyncio
    async def test_predict_is_up():
        from delai.api.fast import app
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/predict", params=test_params)
        assert response.status_code == 200
