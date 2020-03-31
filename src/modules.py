import os 
import logging
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from collections import Counter
from .preprocessing.tools import utils

def convert_to_np(train_data, test_data, target_colname='fraud_ind'):
        y_train = train_data[target_colname].values
        X_train = train_data.drop(columns=target_colname).values
        y_test = test_data[target_colname].values
        X_test = test_data.drop(columns=target_colname).values
        return X_train, y_train, X_test, y_test

class EDA:
    """ EDA 常用工具 """
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.__load_data()

    def __load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # ** 轉換時間
        train_copy = self.train_data.copy()
        test_copy = self.test_data.copy()
        self.train_data['time'] = train_copy.loctm.apply(self.convert_time)
        self.test_data['time'] = test_copy.loctm.apply(self.convert_time)

        self.group_by_target = self.train_data.groupby('fraud_ind')
        self.fraud_records = self.group_by_target.get_group(1)
        self.not_fraud_records = self.group_by_target.get_group(0)

    def hist(self, colname):
        """ train / test hist """
        _ , ax = plt.subplots(1, 2)
        ax[0].set_title('Training set')
        ax[0].hist(self.train_data[colname].dropna())
        ax[0].grid()
        ax[1].set_title('Testing set')
        ax[1].hist(self.test_data[colname].dropna())
        ax[1].grid()
        plt.plot()
    
    def get_num_unique(self, colname, is_show_unique=False):
        """ 該欄位類別數 """
        train_unique = np.unique(self.train_data[colname])
        test_unique = np.unique(self.test_data[colname])
        print('Colname :　{}'.format(colname))
        if is_show_unique:
            print('------------')
            print('Train : ')
            print(train_unique)
            print('------------')
            print('Test : ')
            print(test_unique)
            print('------------')
        print('Number of train unique : {}'.format(train_unique.shape[0]))
        print('Number of test unique  : {}'.format(test_unique.shape[0]))
        return None

    def get_intersection(self, colname):
        """ 回傳該欄位train與test類別的交集 """
        train_set = set(self.train_data[colname])
        test_set = set(self.test_data[colname])
        inter_set = train_set & test_set
        return inter_set

    def get_union(self, colname):
        """ 回傳該欄位train與test類別的聯集 """
        train_set = set(self.train_data[colname])
        test_set = set(self.test_data[colname])
        union_set = train_set | test_set
        return union_set

    def get_num_intersection(self, colname):
        """ 回傳train、test資料的交集種類數 """
        inter_set = self.get_intersection(colname)
        num_inter = len(list(inter_set))
        print('Number of intersection category: {}'.format(num_inter))
        print('Intersection Ratio : ')
        print('Train : {}'.format(num_inter / self.train_data[colname].unique().shape[0]))
        print('Test  : {}'.format(num_inter / self.test_data[colname].unique().shape[0]))
        return None 

    def get_target_ratio(self, colname, interest_set):
        """ 回傳interest_set中有多少筆是盜刷 """
        cnt = 0
        fraud_list = list(self.group_by_target.get_group(1)[colname].values)
        assert type(interest_set) == set
        for val in list(interest_set):
            if val in fraud_list:
                cnt += 1
        print('{} fraud records.'.format(cnt))
        print('fraud rate : {}'.format(cnt / len(fraud_list)))
        return None 

    def get_target_category_by_column(self, colname, top_k=None):
        """ 回傳給定欄位下有盜刷的類別及次數 """
        fraud_list = list(self.group_by_target.get_group(1)[colname].values)
        counter = Counter(fraud_list)
        res = counter.most_common(top_k)
        return res
    
    def check_na(self, colname):
        """ 檢查某欄位是否有na """
        isna_tr = self.train_data[colname].isna().unique()
        isna_te = self.test_data[colname].isna().unique()
        print('Train : ', isna_tr)
        print('Test  : ', isna_te)
        return None

    def convert_time(self, inpuit_values,is_normalize=False):
        denominator = 24 if is_normalize else 1 
        return utils.convert_time(inpuit_values) / denominator

    def group_stat_hist(self, group_colname, stat_colname):
        fig , ax = plt.subplots(3, 2, figsize=(10, 10))
        ax[0, 0].hist(self.fraud_records.groupby(group_colname)[stat_colname].transform('max'), density=True)
        ax[0, 0].set_title('fraud (max)')
        ax[0, 0].grid()
        ax[0, 1].hist(self.not_fraud_records.groupby(group_colname)[stat_colname].transform('max'), density=True)
        ax[0, 1].set_title('not fraud (max)')
        ax[0, 1].grid()
        ax[1, 0].hist(self.fraud_records.groupby(group_colname)[stat_colname].transform('mean'), density=True)
        ax[1, 0].set_title('fraud (maen)')
        ax[1, 0].grid()
        ax[1, 1].hist(self.not_fraud_records.groupby(group_colname)[stat_colname].transform('mean'), density=True)
        ax[1, 1].set_title('not fraud (maen)')
        ax[1, 1].grid()
        ax[2, 0].hist(self.fraud_records.groupby(group_colname)[stat_colname].transform('min'), density=True)
        ax[2, 0].set_title('fraud (min)')
        ax[2, 0].grid()
        ax[2, 1].hist(self.not_fraud_records.groupby(group_colname)[stat_colname].transform('min'), density=True)
        ax[2, 1].set_title('not fraud (min)')
        ax[2, 1].grid()
        plt.plot()

# class SearchProcess:
#     def __init_(self, train_path, preprocess_version='experiments', estimator_version='experiments')

class SubmitProcess:
    def __init__(self, train_path, test_path, preprocess_version='experiments', estimator_version='experiments', output_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.preprocess_version = preprocess_version
        self.estimator_version = estimator_version
        self.output_path = output_path
        self.__load_data()

    def __load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)

    def __process_data(self):
        logging.info('取得前處理版本 : {}'.format(self.preprocess_version))
        preprocesser = PreprocessingCtrl(version=self.preprocess_version)
        logging.info('進行訓練資料前處理.....')
        processed_train = preprocesser.transform(self.train_data)
        logging.info('進行測試資料前處理.....')
        processed_test = preprocesser.transform(self.test_data)
        logging.info('前處理完成.')
        return processed_train, processed_test

    def run(self):
        logging.info('開始提交流程.')
        # ** 取得前處理資料
        processed_train, processed_test = self.__process_data()

        # ** 開始訓練
        logging.info('取得模型版本 : {}'.format(self.estimator_version))
        self.estimator = EstimatorCtrl(self.estimator_version)
        logging.info('開始訓練模型.....')
        self.estimator.fit(processed_train, target_colname='fraud_ind')
        logging.info('訓練完成.')

        # ** 推論&生成提交結果
        logging.info('推論測試資料.....')
        predictions = self.estimator.infer(processed_test)
        predictions_df = pd.DataFrame({'fraud_ind' : predictions})
        predictions_df.index = self.test_data.txkey
        if len(self.output_path) > 0:
            try:
                logging.info('將提交結果存到 : {}'.format(self.output_path))
                predictions_df.to_csv(self.output_path)
            except Exception as e:
                logging.warning('存檔失敗!')
                logging.warning(e)
        logging.info('提交流程結束.')
        return None

class EvaluateProcess:
    def __init__(self, data_path, preprocess_version='experiments', estimator_version='experiments', num_folds=5, save_path=None, load_path=None):
        self.preprocess_version = preprocess_version
        self.estimator = EstimatorCtrl(estimator_version)
        self.data_path = data_path
        self.num_folds = num_folds
        self.save_path = save_path
        self.load_path = load_path
        self.scores = []

    def get_data(self):
        if os.path.exists(self.load_path):
            logging.info('從{}讀取資料.'.format(self.load_path))
            try:
                datasets = self.__load_data()
            except Exception as e:
                logging.warning('讀檔失敗!')
                logging.warning(e)
                logging.info('將重新進行資料前處理.')
                datasets = self.__process_data()
        else:
            datasets =  self.__process_data()
        return datasets

    def __load_data(self, load_path):
        dataset = DataSet()
        train = []
        for index, filename in enumerate(os.listdir(load_path)):
            if not os.path.isdir(filename):
                if filename.find('train_') == -1:
                    pass
                else:
                    train.append(filename)
        for index in range(len(train)):
            for i in range(len(train)):
                if train[index].find(str(i)) == -1:
                    pass
                else:
                    train_data = pd.read_csv(load_path + train[index])
                    test_data = pd.read_csv(load_path + 'test_' + str(index) + '.csv')
                    dataset.add_data(train_data, test_data)
        return dataset

    def __process_data(self):
        logging.info('開始資料前處理流程 : ')
        data = pd.read_csv(self.data_path)
        indices = np.arange(data.shape[0])
        k_fold = KFold(n_splits=self.num_folds, shuffle=True)
        dataset = DataSet()
        logging.info('取得前處理版本 : {}'.format(self.preprocess_version))
        preprocesser = PreprocessingCtrl(version=self.preprocess_version)
        logging.info('將進行{}次OOF前處理.'.format(self.num_folds))
        for i, (train_indices, test_indices) in enumerate(k_fold.split(indices)):
            logging.info('Fold number : {}'.format(i))
            train_data = data.iloc[train_indices, :]
            test_data = data.iloc[test_indices, :]
            processed_train_data = preprocesser.transform(train_data)
            prosessed_test_data = preprocesser.transform(test_data)
            dataset.add_data(processed_train_data, prosessed_test_data)
        if len(self.save_path) > 0:
            try:
                logging.info('正在將處理後的資料存至 : {}....'.format(self.save_path))
                dataset.save(self.save_path)
            except Exception as e:
                logging.warning('存檔失敗')
                logging.warning(e)
        return dataset

    def __evaluate_on_data(self, train_data, test_data, target_colname='fraud_ind'):
        self.estimator.fit(train_data, target_colname=target_colname)
        return self.estimator.evaluate(test_data, target_colname=target_colname)
    
    def run(self):
        logging.info('開始評估流程.')
        datasets = self.get_data()
        logging.info('將進行{} fold交叉驗證 : '.format(datasets.get_fold_number()))
        for i, train_set, test_set in datasets.gen_data():
            logging.info('CV : {}'.format(i))
            score = self.__evaluate_on_data(train_set, test_set)
            logging.info('score : {}'.format(score))
            self.scores.append(score)
        logging.info('交叉驗證結束.')
        logging.info('平均得分 : {}'.format(np.mean(self.scores)))
        logging.info('評估流程結束.')
        return np.mean(self.scores)
        
class DataSet:
    def __init__(self):
        self.train = []
        self.test = []

    def add_data(self, train, test):
        self.train.append(train)
        self.test.append(test)

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        for i, (_train, _test) in enumerate(zip(self.train, self.test)):
            train_path = 'train_' + str(i) + '.csv'
            test_path = 'test_' + str(i) + '.csv'
            _train.to_csv(os.path.join(path, train_path), index=0)
            _test.to_csv(os.path.join(path, test_path), index=0)
        return None

    def gen_data(self):
        for i, (_train, _test) in enumerate(zip(self.train, self.test)):
            yield i, _train, _test

    def get_fold_number(self):
        return len(self.train)

class PreprocessingCtrl:
    def __init__(self, version):
        self.version = version
        self.__get_processer()

    def __get_processer(self):
        if self.version == 'experiments':
            from .preprocessing.experiments import DataPreprocesser   
        elif self.version == 'test':
            from .preprocessing.test import DataPreprocesser
        self.processer = DataPreprocesser()

    def transform(self, data):
        return self.processer.transform(data)

class EstimatorCtrl:
    def __init__(self, version='experiments'):
        self.version = version
        self.__get_estimator()

    def __get_estimator(self):
        if self.version == 'experiments':
            from .estimators.experiments import Estimator 
        elif self.version == 'grid_search_test':
            from .estimators.grid_search_test import Estimator
        self.estimator = Estimator()

    def fit(self, train_data, target_colname):
        self.estimator.fit(train_data, target_colname)

    def evaluate(self, train_data, target_colname):
        return self.estimator.evaluate(train_data, target_colname)

    def get_score(self):
        self.estimator.get_score()

    def infer(self, X):
        return self.estimator.infer(X)

