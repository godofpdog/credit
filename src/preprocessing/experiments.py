import logging
import numpy as np 
import pandas as np 

from .tools import utils
from .tools.encoders import TargetEncoder

class DataPreprocesser:
    def __init__(self):
        pass

    def transform(self, input_data):
        # NOTE define data preprocess
        # NOTE 防止鏈式索引所引發的問題
        data = input_data.copy()

        # target_encoder = TargetEncoder()
        # target_encoding_colnames = []

        """ ** drop cols """
        # NOTE mchno(特店代號) scity(城市) stocn(國家) mcc(mcc code)因為交集類別少所以先捨棄
        drop_colnames = ['acqic', 'bacno', 'cano', 'locdt', 'mchno', 'scity', 'stocn', 'txkey', 'mcc', 'csmcu',
                         'hcefg', 'stscd', 'contp', 'iterm']


        """ 處理類別變數 >> NA """


        """ 轉換時間 """
        converted_toctm = data.loctm.apply(utils.convert_time).values
        data['converted_toctm'] = converted_toctm
        data.drop('loctm', axis=1, inplace=True)

        """ 將多類別特徵 轉為統計量特徵 """
        def get_group_max(input_data, group_colname, stat_colname):
            input_copy = input_data.copy()
            return input_copy.groupby(group_colname)[stat_colname].transform('max')

        def get_group_median(input_data, group_colname, stat_colname):
            input_copy = input_data.copy()
            return input_copy.groupby(group_colname)[stat_colname].transform('median')
        
        data['scity_conam-max'] = get_group_max(data, group_colname='scity', stat_colname='conam')
        data['csmcu_conam-max'] = get_group_max(data, group_colname='csmcu', stat_colname='conam')
        data['acqic_conam-max'] = get_group_max(data, group_colname='acqic', stat_colname='conam')
        data['mchno_conam-max'] = get_group_max(data, group_colname='mchno', stat_colname='conam')
        data['stocn_conam-max'] = get_group_max(data, group_colname='stocn', stat_colname='conam')
        data['bacno_conam-max'] = get_group_max(data, group_colname='bacno', stat_colname='conam')
        data['mcc_conam-max']   = get_group_max(data, group_colname='mcc', stat_colname='conam')

        data['hcefg_conam-max'] = get_group_max(data, group_colname='hcefg', stat_colname='conam')
        data['stscd_conam-max'] = get_group_max(data, group_colname='stscd', stat_colname='conam')
        data['contp_conam-max'] = get_group_max(data, group_colname='contp', stat_colname='conam')
        data['iterm_conam-max'] = get_group_max(data, group_colname='iterm', stat_colname='conam')
        data['ecfg_conam-max']  = get_group_max(data, group_colname='ecfg', stat_colname='conam')
        data['flbmk_conam-max'] = get_group_max(data, group_colname='flbmk', stat_colname='conam')
        data['flg_3dsmk_conam-max']   = get_group_max(data, group_colname='flg_3dsmk', stat_colname='conam')
        data['insfg_conam-max'] = get_group_max(data, group_colname='insfg', stat_colname='conam')
        data['ovrlt_conam-max'] = get_group_max(data, group_colname='ovrlt', stat_colname='conam')
        data['cano_conam-max']  = get_group_max(data, group_colname='cano', stat_colname='conam')


        # ** 重要性不高　先刪除
        data.drop('ecfg', axis=1, inplace=True)
        data.drop('flbmk', axis=1, inplace=True)
        data.drop('flg_3dsmk', axis=1, inplace=True)
        data.drop('insfg', axis=1, inplace=True)
        data.drop('ovrlt', axis=1, inplace=True)

        # data['scity_loctm-median'] = get_group_max(data, group_colname='scity', stat_colname='loctm')
        # data['csmcu_loctm-median'] = get_group_max(data, group_colname='csmcu', stat_colname='loctm')
        # data['acqic_loctm-median'] = get_group_max(data, group_colname='acqic', stat_colname='loctm')
        # data['mchno_loctm-median'] = get_group_max(data, group_colname='mchno', stat_colname='loctm')
        # data['stocn_loctm-median'] = get_group_max(data, group_colname='stocn', stat_colname='loctm')
        # data['bacno_loctm-median'] = get_group_max(data, group_colname='bacno', stat_colname='loctm')
        # data['mcc_loctm-median']   = get_group_max(data, group_colname='mcc', stat_colname='loctm')


        """ """
        data.drop(drop_colnames, axis=1, inplace=True)

        # print(data)

        return data