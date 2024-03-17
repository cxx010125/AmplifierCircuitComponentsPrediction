import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, random_split
from sklearn.tree import DecisionTreeRegressor


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
    print(pca.transform(t))
    return pca.transform(t)


# 10-fold cross validation method
def perform_cross_validation(X, y, learning_rate):
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
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

        # Initialize the decision tree regression model
        tree_reg = DecisionTreeRegressor()

        # Train
        tree_reg.fit(train_features_scaler, train_labels_origin.numpy())

        # Predict
        test_predictions = tree_reg.predict(test_features_scaler)

        # Check the predicted values for each dimension
        target_values = test_labels_origin.clone()
        for i in range(target_values.size(1)):
            # Extract the target and predicted values for the current dimension
            target_dimension = target_values[:, i]
            predictions_dimension = test_predictions[:, i]

            # Calculate relative error: |predicted value - target value| / target value * 100%
            if i == 0:
                relative_error_dim_1.append(100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 1:
                relative_error_dim_2.append(100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 2:
                relative_error_dim_3.append(100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 3:
                relative_error_dim_4.append(100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 4:
                relative_error_dim_5.append(100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))
            elif i == 5:
                relative_error_dim_6.append(100 * abs(target_dimension - predictions_dimension) / abs(target_dimension))

        index = index + 1
    print("Mean dim_1: ", round(np.mean(relative_error_dim_1), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_1)))
    print("Mean dim_2: ", round(np.mean(relative_error_dim_2), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_2)))
    print("Mean dim_3: ", round(np.mean(relative_error_dim_3), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_3)))
    print("Mean dim_4: ", round(np.mean(relative_error_dim_4), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_4)))
    print("Mean dim_5: ", round(np.mean(relative_error_dim_5), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_5)))
    print("Mean dim_6: ", round(np.mean(relative_error_dim_6), 2), "% ,Standard deviation: ",
          round(np.std(relative_error_dim_6)))


if __name__ == '__main__':
    # Read data
    X, y = process_all_files('../data/')
    X = X.reshape(len(X), 3200)

    # PCA
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
    test_features_scaler = scaler.transform(test_features_origin)

    # Tensor again.
    train_features_tensor = torch.tensor(train_features_scaler, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features_scaler, dtype=torch.float32)

    # Setting up input and output dimensions
    input_dim = train_features_tensor.size(1)
    output_dim = test_labels_origin.size(1)
    print("input_dim:", input_dim)
    print("output_dim:", output_dim)

    # Training Decision Tree Models
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_3 = DecisionTreeRegressor(max_depth=10)

    regr_1.fit(train_features_tensor[:, 0].reshape(-1, 1), train_labels_origin[:, 0])
    regr_2.fit(train_features_tensor[:, 0].reshape(-1, 1), train_labels_origin[:, 0])
    regr_3.fit(train_features_tensor[:, 0].reshape(-1, 1), train_labels_origin[:, 0])

    y_1 = regr_1.predict(test_features_tensor[:, 0].reshape(-1, 1))
    y_2 = regr_2.predict(test_features_tensor[:, 0].reshape(-1, 1))
    y_3 = regr_3.predict(test_features_tensor[:, 0].reshape(-1, 1))

    # Sort the test data by the first feature
    sorted_indices = np.argsort(test_features_tensor[:, 0])
    sorted_test_features = test_features_tensor[sorted_indices]
    sorted_test_labels = test_labels_origin[sorted_indices]

    # Corresponding predictions are also sorted
    sorted_y_1 = y_1[sorted_indices]
    sorted_y_2 = y_2[sorted_indices]
    sorted_y_3 = y_3[sorted_indices]

    # Plot
    plt.figure()
    plt.scatter(sorted_test_features[:, 0], sorted_test_labels[:, 0], s=20, edgecolor='black', c="orange", label='Data')
    plt.plot(sorted_test_features[:, 0], sorted_y_1, color="blue", label="max_depth=2", linewidth=2)
    plt.plot(sorted_test_features[:, 0], sorted_y_2, color="red", label="max_depth=5", linewidth=2)
    plt.plot(sorted_test_features[:, 0], sorted_y_3, color="green", label="max_depth=10", linewidth=2)
    plt.xlabel('Data')
    plt.ylabel('Target')
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
