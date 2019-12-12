import numpy as np
import tensorflow as tf

from .metrics import MeanAbsolutePercentageError, TheilU, LinearCorrelation


def build_lstm_model(inputs_shape,
                     layers=5,
                     units=[64, 64, 64, 64, 64],
                     learning_rate=0.05):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(units[0], return_sequences=True, input_shape=inputs_shape))
    for i in range(1, layers):
        if i == layers - 1:
            model.add(tf.keras.layers.LSTM(units[i]))
        else:
            model.add(tf.keras.layers.LSTM(units[i], return_sequences=True))

    model.add(tf.keras.layers.Dense(1))

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
    data = []
    labels = []

    start_index = history_size
    end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])
        labels.append(target[i + target_size])

    return np.array(data), np.array(labels)


def generate_train_val_data(x_train, y_train, x_val, y_val, x_test, y_test,
                            past_history, batch_size):
    # normalization
    data_mean = x_train.mean(axis=0)
    data_std = x_train.std(axis=0)

    x_train = (x_train - data_mean) / data_std
    x_val = (x_val - data_mean) / data_std
    x_test = (x_test - data_mean) / data_std

    # generate time slice
    x_train, y_train = generate_slice_data(x_train, y_train, past_history)
    x_val, y_val = generate_slice_data(x_val, y_val, past_history)
    x_test, y_test = generate_slice_data(x_test, y_test, past_history)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(10000).batch(batch_size).repeat()
    # train_data = train_data.batch(batch_size)

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(batch_size).repeat()

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(batch_size).repeat()

    return train_data, val_data, test_data
