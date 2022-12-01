from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np

def initialize_model():
    '''Initialize the model and return model variable'''
    model = SGDClassifier()
    return model

def compile_model(model):
    '''Needed for if we use a DL model.
    For now, this is blank
    '''
    return model

def train_model(model, X, y,
                batch_size=64, patience=5, validation_split=0.3):
    """
    Fit model and return fitted_model
    IF USING DL, update to:
    Fit model and return a the tuple (fitted_model, history)
    """
    fitted_model = model.partial_fit(X,y)
    return fitted_model

def evaluate_model(model, X_val, y_val):
    '''Pass validation data into model and evaluate performance metrics
    '''

    pass
