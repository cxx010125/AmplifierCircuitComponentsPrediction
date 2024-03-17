import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import math
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, random_split
from sklearn.linear_model import LinearRegression


# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# Processes a single file, returning a list
def process_one_file(file):
    data = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.startswith('!'): # Remove the first 5 rows
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
def do_pca(X, n):
    pca = PCA(n_components = n)
    pca.fit(X)
    print("X.shape after PCA: ", pca.transform(X).shape)
    print(pca.transform(X))
    return pca.transform(X)


# Functions that plot scatterplots and fit straight lines
def plot_scatter_and_fit_line(x, y, label_x, label_y):
    plt.scatter(x, y, label=f'Data points ({label_x}, {label_y})')
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', label='Fitted line')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.show()


# 10-fold cross validation method
def perform_cross_validation(X, y, learning_rate):
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    test_losses = []
    relative_error_dim_1 = []
    relative_error_dim_2 = []
    relative_error_dim_3 = []
    relative_error_dim_4 = []
    relative_error_dim_5 = []
    relative_error_dim_6 = []
    index = 0

    for train_index, test_index in kf.split(X):
        # Segmented data sets
        train_features_origin, test_features_origin = X[train_index], X[test_index]
        train_labels_origin, test_labels_origin = y[train_index], y[test_index]

        # Convert dataset to tensor
        train_features_origin = torch.tensor(train_features_origin, dtype=torch.float32)
        test_features_origin = torch.tensor(test_features_origin, dtype=torch.float32)
        train_labels_origin = torch.tensor(train_labels_origin, dtype=torch.float32)
        test_labels_origin = torch.tensor(test_labels_origin, dtype=torch.float32)

        # Normalization of data (after data segmentation)
        scaler = StandardScaler()
        train_features_scaler = scaler.fit_transform(train_features_origin)
        test_features_scaler = scaler.fit_transform(test_features_origin)

        # PCA
        train_features_scaler = do_pca(train_features_scaler, 10)
        test_features_scaler = do_pca(test_features_scaler, 10)

        # Tensor again.
        train_features_tensor = torch.tensor(train_features_scaler, dtype=torch.float32)
        test_features_tensor = torch.tensor(test_features_scaler, dtype=torch.float32)

        # Setting up input and output dimensions
        input_dim = train_features_tensor.size(1)
        output_dim = test_labels_origin.size(1)
        print("input_dim: ", input_dim)
        print("output_dim: ", output_dim)

        # Train
        model = LinearRegression(input_dim, output_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
        num_epochs = 500
        all_loss = []
        for epoch in range(num_epochs):
            outputs = model(train_features_tensor)
            loss = criterion(outputs, train_labels_origin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss.append(math.log(loss.item()))
            if (epoch + 1) % 100 == 0:
                print(f'Fold:[{index}], Epoch [{epoch + 1}/{num_epochs}], Loss: {math.log(loss.item()):.4f}')

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features_tensor)
            test_loss = criterion(test_outputs, test_labels_origin)
        target_values = test_labels_origin.clone()

        # Check the predicted values for each dimension
        for i in range(target_values.size(1)):
            # Extract target and predicted values for the current dimension (all groups)
            target_dimension = target_values[:, i]
            predictions_dimension = test_outputs[:, i]

            # Calculate relative error: |predicted value - target value| / target value * 100%
            if i == 0:
                relative_error_dim_1.append(
                    100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 1:
                relative_error_dim_2.append(
                    100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 2:
                relative_error_dim_3.append(
                    100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 3:
                relative_error_dim_4.append(
                    100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 4:
                relative_error_dim_5.append(
                    100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 5:
                relative_error_dim_6.append(
                    100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))

        test_losses.append(math.log(test_loss.item()))
        index = index + 1

    print("Mean loss:: ", np.mean(test_losses))
    print("Mean dim_1:: ", round(np.mean(relative_error_dim_1), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_1)))
    print("Mean dim_2:: ", round(np.mean(relative_error_dim_2), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_2)))
    print("Mean dim_3:: ", round(np.mean(relative_error_dim_3), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_3)))
    print("Mean dim_4:: ", round(np.mean(relative_error_dim_4), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_4)))
    print("Mean dim_5:: ", round(np.mean(relative_error_dim_5), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_5)))
    print("Mean dim_6:: ", round(np.mean(relative_error_dim_6), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_6)))


# Plotting Cumulative Variance Contribution Curve
def draw_cumulative(p_target):
    pca = PCA(n_components=10)
    pca.fit(p_target)

    # Obtain the variance explained for each principal component
    explained_variances = pca.explained_variance_ratio_

    # Calculate the cumulative variance explained
    cumulative_variances = np.cumsum(explained_variances)

    # Plotting the cumulative variance explained curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Read data
    X, y = process_all_files('../data/')
    X = X.reshape(len(X), 3200)

    # draw_cumulative(X)

    X = do_pca(X, 10)

    # The result of the current run is to show the fitted curve,
    # if you need to see the prediction range,
    # you need to comment out all the code below '2. Transformation to the torch tensor',
    # and uncomment '1. Perform 10-fold cross validation'.

    # 1. Perform 10-fold cross validation
    # perform_cross_validation(X, y, 0.01)

    # 2. Transformation to the torch tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Define the division ratio of the dataset
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size
    train_dataset, test_dataset = random_split(TensorDataset(X, y), [train_size, test_size])

    # Convert dataset to tensor
    train_features_origin, train_labels_origin = train_dataset[:]
    test_features_origin, test_labels_origin = test_dataset[:]

    # Normalization of data (after data segmentation)
    scaler = StandardScaler()
    train_features_scaler = scaler.fit_transform(train_features_origin)
    test_features_scaler = scaler.fit_transform(test_features_origin)

    # Tensor again
    train_features_tensor = torch.tensor(train_features_scaler, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features_scaler, dtype=torch.float32)

    # Setting up input and output dimensions
    input_dim = train_features_tensor.size(1)
    # output_dim = train_labels_tensor.size(1)
    output_dim = test_labels_origin.size(1)
    print("input_dim:", input_dim)
    print("output_dim:", output_dim)

    # Use only the first output dimension
    # Take the first dimension and convert it to a column vector
    train_labels_tensor = train_labels_origin[:, 0].unsqueeze(1)
    test_labels_tensor = test_labels_origin[:, 0].unsqueeze(1)
    model = LinearRegression(input_dim, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)

    # Train
    losses = []
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(train_features_tensor)
        loss = criterion(outputs, train_labels_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    # Predict
    with torch.no_grad():
        predicted_labels = model(test_features_tensor).numpy()

    # Visualization of fitting curves
    plt.figure(figsize=(8, 6))
    plt.scatter(test_labels_origin[:, 0], predicted_labels[:, 0], color='blue', alpha=0.5)
    plt.xlabel('Predictions')
    plt.ylabel('Target')
    plt.title(' ')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([test_labels_origin[:, 0].min() - 0.5, test_labels_origin[:, 0].max() + 0.5])
    plt.ylim([test_labels_origin[:, 0].min() - 0.5, test_labels_origin[:, 0].max() + 0.5])
    plt.plot([-100, 15000], [-100, 15000], color='red', linewidth=2, linestyle='-')
    plt.tight_layout()
    plt.show()