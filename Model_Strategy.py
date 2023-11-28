# Import Libraries
import tensorflow as tf
import pandas as pd
import os

# Define Helper Functions
def normalize_series(data, min_value, max_value):
    data = (data - min_value) / (max_value - min_value)
    return data

def windowed_dataset(series, batch_size,n_past, n_future, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

#Define the Main Model Function
def solution_model():

    # Reads the dataset.
    data_dir = "sp500_data"

    # Initialize an empty DataFrame
    df_list = []

    # Loop through each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            # Read the CSV file into a DataFrame, selecting only the desired columns
            df2 = pd.read_csv(os.path.join(data_dir, filename), usecols=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        
            # Append the DataFrame to the list
            df_list.append(df2)

    # Concatenate all DataFrames in the list to create a combined DataFrame
    df = pd.concat(df_list, ignore_index=True)

    # Display the combined DataFrame
    print(df.head())

    # Number of features in the dataset. We use all features as predictors to
    # predict all features of future time steps.
    N_FEATURES = len(df.columns) 

    # Selecting Columns and Converting to Numeric
    data = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].apply(pd.to_numeric)
    # Normalizing the Data
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splitting Data into Training and Validation Sets
    SPLIT_TIME = int(len(data) * 0.8) 
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    # TensorFlow Setup
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)

    # Setting Batch Size
    BATCH_SIZE = 128  


    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 10

    # Number of future time steps which are to be predicted.
    N_FUTURE = 10 

    # By how many positions the window slides to create a new window
    SHIFT = 1  

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Define and Compile the Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(N_PAST, N_FEATURES))),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        #tf.keras.layers.Lambda(lambda x: x * 400),
        tf.keras.layers.Dense(1),
    ])

    # Code to compile the model
    model.compile(loss=tf.keras.losses.Huber(), 
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=["mae"])
    
    # Implement learning rate scheduler
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.01)

    # Implement callbacks
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Implement model checkpointing
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

    # Model fit
    model.fit(train_set, epochs=80, validation_data=valid_set,
              callbacks=[lr_callback,model_checkpoint, early_stopping, model_checkpoint])

    return model

if __name__ == '__main__':
    model = solution_model()
    # Save the Model
    model.save("mymodel.h5")
