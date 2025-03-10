import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

import math

from pygam import LinearGAM, s

def evaluate_model(actual, predicted, dataset_name=""):
    """
    Evaluate a model's performance and return metrics in a dictionary.
    
    Parameters:
    -----------
    actual : array-like
        The actual values
    predicted : array-like
        The predicted values
    dataset_name : str, optional
        Name of the dataset (e.g., "Training", "Test")
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    accuracy = 1 - (rmse / np.mean(actual))
    
    metrics = {
        "Dataset": dataset_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R^2": r2,
        "Accuracy": accuracy,
        "Accuracy (%)": accuracy * 100
    }
    
    return metrics

def plot_predictions(actual_train, pred_train, actual_test, pred_test, 
                    train_dates, test_dates, model_name="Model"):
    """
    Plot actual vs predicted values for both training and test sets.
    
    Parameters:
    -----------
    actual_train : array-like
        Actual values from training set
    pred_train : array-like
        Predicted values for training set
    actual_test : array-like
        Actual values from test set
    pred_test : array-like
        Predicted values for test set
    train_dates : array-like
        Dates corresponding to training data
    test_dates : array-like
        Dates corresponding to test data
    model_name : str, optional
        Name of the model for plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(train_dates, actual_train, color='blue', label='Actual (Train)')
    plt.plot(train_dates, pred_train, color='green', linestyle='--', label='Predicted (Train)')
    
    # Plot test data
    plt.plot(test_dates, actual_test, color='blue', label='Actual (Test)')
    plt.plot(test_dates, pred_test, color='red', linestyle='--', label='Predicted (Test)')
    
    # Add a vertical line to separate train and test sets
    split_date = test_dates.iloc[0] if hasattr(test_dates, 'iloc') else test_dates[0]
    plt.axvline(x=split_date, color='black', linestyle='-', alpha=0.3)
    plt.text(split_date, plt.ylim()[1]*0.9, 'Train-Test Split', 
             horizontalalignment='center', backgroundcolor='white')
    
    plt.title(f'{model_name}: Unemployment Rate - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def display_metrics_table(train_metrics, test_metrics, model_name="Model"):
    """
    Display a formatted table comparing training and test metrics.
    
    Parameters:
    -----------
    train_metrics : dict
        Dictionary containing training metrics
    test_metrics : dict
        Dictionary containing test metrics
    model_name : str, optional
        Name of the model for the table header
    """
    print(f"\nModel Evaluation: {model_name}")
    print("-" * 60)
    metrics_names = ["MSE", "RMSE", "MAE", "R^2", "Accuracy", "Accuracy (%)"]
    print(f"{'Metric':<15} {'Training Set':<20} {'Test Set':<20}")
    print("-" * 60)
    
    for metric in metrics_names:
        if metric == "Accuracy (%)":
            print(f"{metric:<15} {train_metrics[metric]:.2f}% {' '*12} {test_metrics[metric]:.2f}%")
        else:
            print(f"{metric:<15} {train_metrics[metric]:.4f} {' '*14} {test_metrics[metric]:.4f}")
    
    print("-" * 60)