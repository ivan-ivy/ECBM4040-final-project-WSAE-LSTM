# ECBM E4040 Final Project: WSAE-LSTM
# Author: Yifan Liu
# This is a utility function to help you download the dataset and preprocess the data we use for this homework.
# requires several modules: _pickle, tarfile, glob. If you don't have them, search the web on how to install them.
# You are free to change the code as you like.

# import modules
import os
import urllib.request as url

import pandas as pd

# define absolute path, to make sure use this function in jupyter notebook will not 
# casue a problem
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../../data/raw"))

# for training convenience
INDEX_SHEET_NAME = ['HangSeng Index Data',
                    'S&P500 Index Data',
                    'CSI300 Index Data',
                    'DJIA index Data',
                    'Nikkei 225 index Data',
                    'Nifty 50 index Data'
                    ]


def download_data():
    """Download the Raw data xlsx from the website provided by the author. The
    data file will be store in the ../../data/raw folder.
    :return: None
    """

    if not os.path.exists(RAW_DATA_DIR + '/RawData.xlsx'):
        os.makedirs(RAW_DATA_DIR)
        print('Start downloading data...')
        url.urlretrieve(url=r"https://ndownloader.figshare.com/files/8493140",
                        filename=RAW_DATA_DIR + '/RawData.xlsx')
        print('Download complete.')
    else:
        print('Data already exists.')


def load_data(sheet_name="HangSeng Index Data"):
    """load the data set of specific type. :param sheet_name: Specify the data.
    :return: A data frame contains the data of certain index.

    Args:
        sheet_name: specify the sheet name to load data
    """
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists(RAW_DATA_DIR + '/RawData.xlsx'):
        download_data()

    return pd.read_excel(RAW_DATA_DIR + '/RawData.xlsx', sheet_name=sheet_name)
