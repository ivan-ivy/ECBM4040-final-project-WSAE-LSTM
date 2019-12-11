import os

import numpy as np
import pandas as pd

NUM_TRAIN = 24
NUM_VAL = 3
NUM_TEST = 3


def generate_features(raw: pd.DataFrame, sheet_name):
    month_lst = list(set(raw.Ntime // 100))
    month_lst.sort()

    def get_index(keys):
        ind_lst = []
        for key in keys:
            ind_lst.extend(index_dict[key])
        return ind_lst

    index_dict = dict()
    for month in month_lst:
        index_dict[month] = raw[raw.Ntime // 100 == month].index.to_list()

    save_dir = f'../data/interim/{sheet_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(0, len(month_lst) - NUM_TRAIN - NUM_VAL - NUM_TEST + 3, 3):
        train_ind = get_index(month_lst[i:i + NUM_TRAIN])
        val_ind = get_index(month_lst[i + NUM_TRAIN:i + NUM_TRAIN + NUM_VAL])
        test_index = get_index(month_lst[i + NUM_TRAIN + NUM_VAL:i + NUM_TRAIN + NUM_VAL + NUM_TEST])

        save_dir = f'../data/interim/{sheet_name}/{month_lst[i + NUM_TRAIN + NUM_VAL]}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        np.save(file=save_dir + '/X_train.npy',
                arr=raw.iloc[train_ind, 2:].values)

        # save the second column 'closing price' as target for training LSTM and RNN
        np.save(file=save_dir + '/Y_train.npy',
                arr=raw.iloc[train_ind, 2].values)

        np.save(file=save_dir + '/X_val.npy',
                arr=raw.iloc[val_ind, 2:].values)
        np.save(file=save_dir + '/Y_val.npy',
                arr=raw.iloc[val_ind, 2].values)

        np.save(file=save_dir + '/X_test.npy',
                arr=raw.iloc[test_index, 2:].values)
        np.save(file=save_dir + '/Y_test.npy',
                arr=raw.iloc[test_index, 2].values)

        # print(f"{sheet_name}: {month_lst[i + NUM_TRAIN + NUM_VAL]} generated!")

    print(">>>> Feature generation complete! <<<<")
