import os
import logging
import numpy as np
import pandas as pd 

from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO)

class TargetEncoder:
    def __init__(self, target_statisic='mean', min_samples_leaf=5, smoothing_weight=1, missing_handler='value', unkown_handler='value'):
        self.target_statisic = target_statisic
        self.min_samples_leaf = min_samples_leaf
        self.smoothing_weight = smoothing_weight
        self.missing_handler = missing_handler
        self.unkown_handler = unkown_handler
        self.__check_arguments()

    def __check_arguments(self):
        assert self.target_statisic in ('mean', 'median', 'count', 'var'), 'Argument `target_statisic` must in (`mean`, `median`, `count`, `var`)'
        assert self.missing_handler in ('value', 'drop'), 'Argument `missing_handler` must in (`value`, `drop`)'
        assert self.unkown_handler in ('value', 'drop'), 'Argument `unkown_handler` must in (`value`, `drop`)'
        # if self.folds:
        #     assert type(self.folds) in (int, list), 'type of folds must to be `int` or `list`'
        #     if type(self.folds) == list and len(self.folds) > 2: raise ValueError('length of `folds` must less than 2')

    def fit(self, encode_data, target_values):
        assert type(encode_data) == pd.Series, 'Data type error.'
        assert type(target_values) == pd.Series, 'Data type error.'
        assert len(encode_data) == len(target_values), 'Length not the same error.'

        self.target_name = target_values.name
        self.statistics, self.prior = self.__get_target(encode_data, target_values)
        return self

    def transform(self, input_data):
        out_data = self.__transform(self.statistics, self.prior, input_data)
        return out_data

    def oof_encoding(self, encode_data, target_values, num_folds):
        # NOTE only for training
        self.target_name = target_values.name
        indices = np.arange(encode_data.shape[0])
        k_fold = KFold(n_splits=num_folds)
        out_data = encode_data.copy()
        for i, (inf_indices, oof_indices) in enumerate(k_fold.split(indices)):
            # logging.debug('fold : {}'.format(i))
            inf_targets = target_values.iloc[inf_indices]
            inf_data = encode_data.iloc[inf_indices]
            oof_data = encode_data.iloc[oof_indices]
            inf_encode, prior = self.__get_target(inf_data, inf_targets)
            out_data.iloc[oof_indices] = self.__transform(inf_encode, prior, oof_data).values.reshape(-1)
        return out_data

    def __transform(self, encode_statistics, prior, encode_data, is_fillna=True):
        left = encode_statistics.reset_index().rename(columns = {"index" : self.target_name, self.target_name: "encode"})
        right = encode_data.to_frame(encode_data.name)
        res = pd.merge(left, right, on=encode_data.name, how='right', sort=False).fillna(prior)
        return res.drop(encode_data.name, axis=1)

    def __get_target(self, encode_data, target_values):
        # TODO prior 需增加其他統計量
        concate_data = pd.concat([encode_data, target_values], axis=1)
        statistics = concate_data.groupby(encode_data.name)[target_values.name].agg([self.target_statisic, 'count'])
        n = statistics['count']
        prior = target_values.mean()
        post = statistics[self.target_statisic]
        _lambda = 1 / (1 + np.exp((-1) * (n - self.min_samples_leaf) / self.smoothing_weight))
        statistics['encode'] = _lambda * post + (1 - _lambda) * prior
        statistics.drop(['mean', 'count'], axis=1, inplace=True)
        return statistics, prior




