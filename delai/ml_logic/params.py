import os
import numpy as np

DATASET_SIZE = os.environ.get("DATASET_SIZE")
VALIDATION_DATASET_SIZE = os.environ.get("VALIDATION_DATASET_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")

COLUMN_NAMES_RAW = ["Unnamed: 0"
,"FlightDate"
,"Year"
,"Quarter"
,"Month"
,"DayofMonth"
,"DayOfWeek"
,"Airline"
,"Operating_Airline"
,"Marketing_Airline_Network"
,"Flight_Number_Marketing_Airline"
,"Origin"
,"Dest"
,"CRSDepTime"
,"OriginAirportID"
,"OriginCityName"
,"OriginStateName"
,"DestAirportID"
,"DestCityName"
,"DestStateName"
,"CRSArrTime"
,"Distance"
,"ArrDelayMinutes"
,"Cancelled"
,"Diverted"]

################## VALIDATIONS #################

env_valid_options = dict(
    DATASET_SIZE=["1k", "10k", "100k", "500k", "50M", "new", "full"],
    VALIDATION_DATASET_SIZE=["1k", "10k", "100k", "500k", "500k", "new"],
    DATA_SOURCE=["local", "big query"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
