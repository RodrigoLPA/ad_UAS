# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from tqdm import tqdm
import sys
import xgboost as xgb
import time
import shap
import os
import math
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# %%
# ML
import torch
import torch.nn as nn
import torch.optim as optim

# Anomaly detection models
import pyod
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

# ==================================================================================================
# Anomaly Detection Methods
# ==================================================================================================
# INPUTS: X_train, X_test
# OUTPUTS: y_pred, y_scores


# OCSVM training and prediction
def ocsvm_train_and_predict(
    X_train,
    X_test,
    kernel="rbf",
    nu=0.5,
    contamination=0.1,
    verbose=False,
    print_time=False,
):
    """
    Trains an One-Class SVM model on the given training data and predicts the labels and anomaly scores for the test data.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        kernel (str, optional): Kernel function for the SVM. Defaults to "rbf".
        nu (float, optional): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Defaults to 0.5.
        contamination (float, optional): The proportion of outliers in the data set. Defaults to 0.1.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.

    Returns:
        y_pred (array-like): Predicted labels for the test data.
        y_scores (array-like): Anomaly scores for the test data.
    """
    if time:
        start_time = time.time()

    # Function code here

    ## Training the model
    model = OCSVM(kernel=kernel, nu=nu, contamination=contamination, verbose=verbose)
    model.fit(X_train)

    ## Predicting
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)

    if time:
        end_time = time.time()
        print(f"Execution time: {(end_time - start_time)/60} minutes.")

    return y_pred, y_scores


# DeepSVDD training and prediction
def deepsvdd_train_and_predict(
    X_train,
    X_test,
    # c="forward_nn_pass",
    use_ae=False,
    hidden_neurons=[64, 32],
    hidden_activation="relu",
    output_activation="sigmoid",
    optimizer="adam",
    epochs=100,
    batch_size=32,
    dropout_rate=0.2,
    l2_regularizer=0.1,
    validation_size=0.1,
    preprocessing=True,
    verbose=1,
    random_state=None,
    contamination=0.1,
    print_time=False,
):
    """
    Trains a Deep SVDD model on the given training data and predicts the labels and anomaly scores for the test data.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        c (str, optional): Center initialization method. Defaults to "forward_nn_pass".
        use_ae (bool, optional): Whether to use autoencoder for pre-training. Defaults to False.
        hidden_neurons (list, optional): Number of neurons in each hidden layer. Defaults to [64, 32].
        hidden_activation (str, optional): Activation function for hidden layers. Defaults to "relu".
        output_activation (str, optional): Activation function for output layer. Defaults to "sigmoid".
        optimizer (str, optional): Optimization algorithm. Defaults to "adam".
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        l2_regularizer (float, optional): L2 regularization coefficient. Defaults to 0.1.
        validation_size (float, optional): Proportion of training data to use for validation. Defaults to 0.1.
        preprocessing (bool, optional): Whether to apply preprocessing to the data. Defaults to True.
        verbose (int, optional): Verbosity level. Defaults to 1.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        contamination (float, optional): The proportion of outliers in the data set. Defaults to 0.1.

    Returns:
        y_pred (array-like): Predicted labels for the test data.
        y_scores (array-like): Anomaly scores for the test data.
    """
    if time:
        start_time = time.time()
    ## Training the model
    model = DeepSVDD(
        # c=c,
        use_ae=use_ae,
        hidden_neurons=hidden_neurons,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        l2_regularizer=l2_regularizer,
        validation_size=validation_size,
        preprocessing=preprocessing,
        verbose=verbose,
        random_state=random_state,
        contamination=contamination,
    )
    model.fit(X_train)

    ## Predicting
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)

    if time:
        end_time = time.time()
        print(f"Execution time: {(end_time - start_time)/60} minutes.")
    return y_pred, y_scores, model.history_


# ==================================================================================================
# Feature attribution distribution methods
# ==================================================================================================
def supervised_task_xgb(X_train, y_train, X_test, y_test, task_type="classification"):
    if task_type == "classification":
        xgb_model = xgb.XGBClassifier()
    elif task_type == "regression":
        xgb_model = xgb.XGBRegressor()

    ## Train xgboost
    xgb_model.fit(X_train, y_train)

    ## Predict on the test data
    y_pred = xgb_model.predict(X_test)

    ## Compute Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    ## Compute Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    ## Compute R-squared
    r2 = r2_score(y_test, y_pred)

    ## Printing the metrics
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return xgb_model


# ===================================================================================================
# Feature attribution methods
def shap_explanation(trained_xgb_model, X_test):
    # Extracting the feature attributions
    ## Initialize the SHAP explainer
    explainer = shap.Explainer(trained_xgb_model)

    # Compute the Shapley values for the test set
    shap_values = explainer(X_test)

    # Feature attributions as new numpy array
    explainer_array = shap_values.values

    return explainer_array
