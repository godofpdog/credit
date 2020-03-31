import lightgbm as lgb
from .tools.utils import LGB_GridSearchCV, lgb_f1_score
from sklearn.metrics import f1_score, auc
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

""" 在提交流程中測試grid search CV """

class Estimator:
    def __init__(self):
        pass 

    def fit(self, train_data, target_colname):

        # ** 轉換資料
        X = train_data.drop(columns=target_colname)
        y = train_data[target_colname]
        lgb_data = lgb.Dataset(X, label=y, free_raw_data=False)

        # NOTE 用grid search方法調參數
        params = {
            'max_depth':5, 
            'learning_rate':0.1,
            'num_leaves':63,
            'feature_fraction':0.8,
            'bagging_fraction':0.8,
            'min_split_gain':0,
            'bagging_freq':1,
            'objective':'binary', 
            'num_threads':4, 
            'boost_from_average':False,
            'min_data_in_leaf':20,
            'is_unbalance':True,
            'verbose':-1,
            }

        # params_grid = {
        #     'max_depth':[-1, 5],
        #     'num_leaves':[63],
        #     'feature_fraction':[0.9],
        #     'bagging_freq':[1],
        # }

        # # ** 初始化 LGB_GridSearchCV_old
        # lgb_cv = LGB_GridSearchCV(params, params_grid, nfold=5, num_boost_round=500, early_stop=10)
        # lgb_cv.fit(lgb_data)

        # ** 取得CV最佳參數
        # num_round = lgb_cv.get_boost_rounds()
        # best_params = lgb_cv.get_best_params()

        # ** re-train
        self.bst = lgb.train(params, lgb_data, 400, valid_sets=[lgb_data], feval=lgb_f1_score, verbose_eval=100, early_stopping_rounds=10)
        # plt.figure(figsize=(12,6))
        # lgb.plot_importance(self.bst, max_num_features=30)
        # plt.title("Featurertances")
        # plt.show()


    def evaluate(self, test_data, target_colname):
        """ 回傳驗證評估分數 """

        # ** 轉換資料
        X = test_data.drop(columns=target_colname)
        y = test_data[target_colname]

        threshold = .5
        bst_predictions = 1 * (self.bst.predict(X.values) > threshold)
        print('set >> ', set(y.values) - set(bst_predictions))
        self.score = f1_score(y.values, bst_predictions)
        return self.score

    def get_score(self):
        return self.score

    def infer(self, X):
        threshold = 0.5
        return 1 * (self.bst.predict(X) > threshold)