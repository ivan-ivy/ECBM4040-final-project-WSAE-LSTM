import os
import pickle
import sys

import pandas as pd

sys.path.append("../")
from src.data.make_dataset import INDEX_SHEET_NAME

MODEL_NAME = ['lstm', 'wlstm', 'wsae-lstm', 'sae-lstm']
FILE_NAME = '_train_result.pickle'


def get_pivot_table(index_name):
    """
    Args:
        index_name: specify the target data
    """
    result = dict()
    for model in MODEL_NAME:
        result[model] = dict()
        for sheet in INDEX_SHEET_NAME:
            path = os.path.join(model, sheet + FILE_NAME)
            with open(path, 'rb') as handler:
                temp_dict = pickle.load(handler)
                for k in temp_dict.keys():
                    temp_dict[k] = {'mape': temp_dict[k][2],
                                    'r': temp_dict[k][3],
                                    'theil_u': temp_dict[k][4]}
                result[model][sheet] = temp_dict

    result_df = pd.DataFrame()

    for model in MODEL_NAME:
        df = pd.DataFrame(result[model][index_name]).T.reset_index()
        df = df.groupby(df['index'].map(lambda x: x[:4])).mean().reset_index()
        df.columns = ['Year', 'MAPE', 'R', 'Theil U']
        df['Model'] = model.upper()
        result_df = pd.concat([result_df, df])

    result_table = result_df.pivot_table(index=['Year'], columns=['Model']).drop('2016')
    result_table.loc['Average'] = result_df.pivot_table(index=['Year'], columns=['Model']).drop('2016').mean(axis=0)
    return result_table
