import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from joblib import Parallel, delayed


# Processes a single file, returning a list
def process_one_file(file):
    data = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.startswith('!'):
                temp = []
                params = line.split('&')
                if(len(params) != 9):
                    params = line.split('\t')
                    if (len(params) != 9):
                        print("*************")
                for i in range(len(params)):
                    if i % 2 == 1:
                        temp.append(float(params[i]))
                data.append(temp)
    return data


# Consolidate all files in the destination folder (read files based on prefix)
def process_all_files(directory):
    all_data = []
    all_target = []
    entries = os.listdir(directory)
    for file_name in entries:
        full_path = os.path.join(directory, file_name)
        if os.path.isfile(full_path):
            print(f"-------------File: {full_path}-------------")
            nums =file_name.split(".tx")[0].split("_")
            target = []
            for num in nums:
                target.append(float(num))
            all_target.append(target)
            file_now = process_one_file(full_path)
            all_data.append(file_now)
    all_data = np.array(all_data)
    all_target = np.array(all_target)
    return all_data, all_target


# Reduce the dimensionality of the whole array
def do_pca(t, n):
    pca = PCA(n_components=n)
    pca.fit(t)
    print("X.shape after PCA: ", pca.transform(t).shape)
    return pca.transform(t)


def plot_predictions(test_labels, test_predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(test_labels, test_predictions, color='blue')
    plt.plot(test_labels, test_labels, color='red', linestyle='--')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('True vs Predicted values')
    plt.grid(True)
    plt.show()


def plot_fit_curve(train_labels, train_features, svr_models):
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(svr_models):
        plt.scatter(train_features[:, 0], train_labels[:, i], label=f'Dimension {i+1}')
        plt.plot(train_features[:, 0], model.predict(train_features), label=f'Fit Dimension {i+1}')
    plt.xlabel('Feature 1')
    plt.ylabel('Labels')
    plt.title('SVR Fit Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


# 10-fold cross validation method
def perform_cross_validation(X, y, learning_rate):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    relative_errors = [[] for _ in range(y.shape[1])]
    for train_index, test_index in kf.split(X):
        # Segmented data sets
        train_features, test_features = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]

        # Standardization of data
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # PCA
        train_features = do_pca(train_features, 2)
        test_features = do_pca(test_features, 2)

        # Convert to a PyTorch tensor and move it to the GPU
        train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).to(device)
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)

        # Initialize the list of SVR models, one per output dimension
        svr_models = [SVR(kernel='rbf') for _ in range(train_labels.shape[1])]

        # Training each SVR model
        for i in range(train_labels.shape[1]):
            svr_models[i].fit(train_features_tensor.cpu().numpy(), train_labels[:, i])

        # Prediction using a trained model
        test_predictions = np.array([model.predict(test_features_tensor.cpu().numpy()) for model in svr_models]).T

        # Calculation of relative error
        for i in range(train_labels.shape[1]):
            relative_error = 100 * np.mean(
                np.abs(test_labels[:, i] - test_predictions[:, i]) / np.abs(test_labels[:, i]))
            relative_errors[i].append(relative_error)

    # Output Mean Relative Error
    for i, errors in enumerate(relative_errors):
        # print(errors)
        print(f"Mean relative error dim_{i + 1}: {round(np.mean(errors), 2)}%  {np.std(errors)}")

    # Visualization of forecasts
    plot_predictions(test_labels, test_predictions)

    # Visualization of fitting curves
    plot_fit_curve(train_labels, train_features, svr_models)


if __name__ == '__main__':
    # Check for available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Read data
    X, y = process_all_files('../data/')
    X = X.reshape(len(X), 3200)

    # Perform 10-fold cross validation
    perform_cross_validation(X, y, 0.01)
