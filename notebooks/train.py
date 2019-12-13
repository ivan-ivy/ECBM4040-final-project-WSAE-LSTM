import logging
import os
import pickle
import sys
import numpy as np

# set log level to ignore warning due to bugs of tensorflow 2.0
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


sys.path.append("../")
from src.data.make_dataset import INDEX_SHEET_NAME
from src.models.LSTM import build_lstm_model, generate_train_val_data

if __name__ == '__main__':
    result_dict = dict()
    EPOCHS = 6000
    past_history = 4

    for index in INDEX_SHEET_NAME:
        print(f"Start {index} part!")
        result_dict[index] = dict()

        data_dir = f'../data/processed/wsae/{index}'
        train_lst = os.listdir(data_dir)
        for name in train_lst:
            x_train = np.load(data_dir + f'/{name}/X_train.npy')
            y_train = np.load(data_dir + f'/{name}/Y_train.npy')
            x_val = np.load(data_dir + f'/{name}/X_val.npy')
            y_val = np.load(data_dir + f'/{name}/Y_val.npy')
            x_test = np.load(data_dir + f'/{name}/X_test.npy')
            y_test = np.load(data_dir + f'/{name}/Y_test.npy')

            train_data, val_data, test_data = generate_train_val_data(
                x_train, y_train, x_val, y_val, x_test, y_test,
                past_history=4, batch_size=60
            )

            lstm = build_lstm_model(inputs_shape=[4, 10],
                                    layers=5,
                                    units=[64, 64, 64, 64, 64],
                                    learning_rate=0.05)
            lstm.fit(train_data,
                     epochs=EPOCHS,
                     steps_per_epoch=(y_train.shape[0] // 60),
                     validation_data=val_data,
                     validation_steps=1,
                     verbose=0)

            result_dict[index][name] = lstm.evaluate(test_data, steps=1)

            save_dir = f'../models/{index}/{name}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            lstm.save(save_dir + '/wase-lstmt.h5')
            print(f">>>>{index} {name} done!<<<<")

        with open(f'{index}_train_result.pickle', 'wb') as handle:
            pickle.dump(result_dict[index], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'train_result.pickle', 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

