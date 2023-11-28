# Importing Libraries
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Normalization Function
def normalize_series(data, min_value, max_value):
    data = (data - min_value) / (max_value - min_value)
    return data

# Denormalization Function
def denormalize_series(normalized_data, min_value, max_value):
    original_data = normalized_data * (max_value - min_value) + min_value
    return original_data

# Windowed Dataset Function
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

# Loading the Pre-trained Model
model = tf.keras.models.load_model('mymodel.h5')

# Load the CSV file and select the columns of interest
df = pd.read_csv(file_path, usecols=["Open", "High", "Low", "Close", "Adj Close", "Volume"])

N_FEATURES = len(df.columns)

# Normalizes the data
data = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].apply(pd.to_numeric)
data_min = data.min(axis=0)
data_max = data.max(axis=0)
data = normalize_series(data, data_min, data_max)

N_PAST = 10
N_FUTURE = 10
SHIFT = 1
BATCH_SIZE = 128

# Split the data into training and testing sets
train_data = data.iloc[:len(data) // 2]  # Use the first half for training
test_data = data.iloc[len(data) // 2:]  # Use the second half for testing

train_data = windowed_dataset(train_data, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)
test_data = windowed_dataset(test_data, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)

# Convert the data to a NumPy array of float32
train_input_data = np.concatenate([x.numpy() for x, _ in train_data])
test_input_data = np.concatenate([x.numpy() for x, _ in test_data])

# Make predictions using the loaded model
predictions_normalized = model.predict(test_input_data)

# Denormalize the predictions individually for each feature
predictions = np.zeros_like(predictions_normalized)
for i in range(N_FEATURES):
    predictions[:, i] = denormalize_series(predictions_normalized[:, i], data_min[i], data_max[i])

# Ensure that test_data and predictions have the same number of samples
test_data_np = np.concatenate([y.numpy() for _, y in test_data])
predictions_np = predictions

# Check the shapes of the arrays
print("Shape of test_data_np:", test_data_np.shape)
print("Shape of predictions_np:", predictions_np.shape)

if len(test_data_np.shape) == 3:
    test_data_np = test_data_np[:, :, 0]  
if len(predictions_np.shape) == 3:
    predictions_np = predictions_np[:, :, 0]  

# Calculate Mean Absolute Error (MAE) using NumPy arrays
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test_data_np, predictions_np)
print("Mean Absolute Error:", mae)



