import numpy as np
import tensorflow as tf

from .metrics import MeanAbsolutePercentageError, TheilU, LinearCorrelation


def build_lstm_model(inputs_shape,
                     layers=5,
                     units=[64, 64, 64, 64, 64],
                     learning_rate=0.05):
    """
    Args:
        inputs_shape: shape of inputs
        layers: number of layers
        units: a list of number of hidden units of each layer
        learning_rate: learning rate
    """
    model = tf.keras.Sequential()

    # ad layers using loop
    model.add(tf.keras.layers.LSTM(units[0], return_sequences=True, input_shape=inputs_shape))
    for i in range(1, layers):
        if i == layers - 1:
            model.add(tf.keras.layers.LSTM(units[i]))
        else:
            model.add(tf.keras.layers.LSTM(units[i], return_sequences=True))

    model.add(tf.keras.layers.Dense(1))

    # add custom metrics, and use tf.keras.metrics.MeanAbsolutePercentageError as validation
    model.compile(loss='mse',
                  optimizer='Adam',
                  metrics=[
                      tf.keras.metrics.MeanAbsolutePercentageError(),
                      MeanAbsolutePercentageError(),
                      LinearCorrelation(),
                      TheilU()],
                  lr=learning_rate
                  )

    return model


def generate_slice_data(dataset,
                        target,
                        history_size,
                        target_size=0):
    """
    Args:
        dataset: inputs data
        target: label of data
        history_size: time window of prediction
        target_size: steps of prediction, default 0 for one-step-ahead prediction
    """
    data = []
    labels = []

    # set start and end index
    start_index = history_size
    end_index = len(dataset) - target_size

    # make slice
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])
        labels.append(target[i + target_size])

    return np.array(data), np.array(labels)


def generate_train_val_data(x_train, y_train, x_val, y_val, x_test, y_test,
                            past_history, batch_size):
    # normalization
    """
    Args:
        x_train: training set
        y_train: training label
        x_val: validation set
        y_val: validation label
        x_test: test set
        y_test: test label
        past_history: time window for prediction
        batch_size: batch size
    """
    data_mean = x_train.mean(axis=0)
    data_std = x_train.std(axis=0)
    x_train = (x_train - data_mean) / data_std

    # normalize validating set and test set using the training set
    x_val = (x_val - data_mean) / data_std
    x_test = (x_test - data_mean) / data_std

    # generate time slice
    x_train, y_train = generate_slice_data(x_train, y_train, past_history)
    x_val, y_val = generate_slice_data(x_val, y_val, past_history)
    x_test, y_test = generate_slice_data(x_test, y_test, past_history)

    # use tf.data to generate batch
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(10000).batch(batch_size).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(batch_size).repeat()

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(batch_size).repeat()

    return train_data, val_data, test_data
