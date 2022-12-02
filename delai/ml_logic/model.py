from sklearn.linear_model import SGDClassifier
from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple

import pandas as pd
import numpy as np

def initialize_model(X):
    '''Initialize the model and return model variable'''
    reg = regularizers.l1_l2(l2=0.005)

    model = Sequential()
    model.add(layers.BatchNormalization(input_shape=X.shape[1:]))
    model.add(layers.Dense(20, activation="relu", kernel_regularizer=reg))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def compile_model(model):
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam()
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    print("\n✅ model compiled")

    return model

def train_model(model, X, y,
                batch_size=64,
                patience=5,
                validation_split=0.3,
                validation_data=None):
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print("Train model...")

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history

def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(f"\nEvaluate model on {len(X)} rows...")

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    f1 = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(f1, 2)}")

    return metrics


def test_model_run(X, y):
    model = initialize_model(X)
    model = compile_model(model)
    return train_model(model, X, y)
