""" utils functions for data pre-processing """

import numpy as np 
import pandas as pd 
import logging

def get_intersection_category(data_frame, colname):
    """ 回傳欄位下train、test交集的類別"""
    pass 

import logging
import numpy as np 
import pandas as pd 

def binary_lebel_encoding(input_value):
    if input_value in ('Y', 'y'):
        out = 1
    elif input_value in ('N', 'n'):
        out = 0
    else:
        out = 2
    return out

def process_target_encoding_train(input_data, encoder, colnames, target_name='fraud_ind', n_folds=3):
    # logging.info('start create target encoding features.')
    data = input_data.copy()
    for colname in colnames:
        feature_name = colname + '_target_encoding'
        # logging.debug('creating new feature : {}...'.format(feature_name))
        # print('******************************')
        encoding_feature = encoder.oof_encoding(data[colname], data[target_name], n_folds).values
        data[feature_name] = encoding_feature.copy()
        # data.assign(feature_name=encoding_feature.copy())
        # logging.debug('delete original feature : {}.'.format(colname))
        data.drop(colname, axis=1, inplace=True)
    return data

def process_target_encoding_test():
    pass

def split_time(input_value):
    hrs = int(input_value // 1e4)
    mins = int(input_value % 1e4 // 1e2)
    sec = int(input_value % 1e2 / 1)
    return hrs, mins, sec

def convert_time(input_value, digits=4):
    hrs, mins, sec = split_time(input_value)
    time = hrs + mins / 60 + sec / 3600
    return int(round(time, digits)) 
