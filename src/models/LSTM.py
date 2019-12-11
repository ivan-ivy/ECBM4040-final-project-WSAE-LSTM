import numpy as np
import tensorflow as tf


def build_lstm_model(inputs_shape,
                     layers=5,
                     units=[64, 64, 64, 64, 64]):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(units[0], return_sequences=True, input_shape=inputs_shape))
    for i in range(1, layers):
        model.add(tf.keras.layers.LSTM(units[i], return_sequences=True))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='mse',
                  optimizer='Adam',
                  metrics=['accuracy'])

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
