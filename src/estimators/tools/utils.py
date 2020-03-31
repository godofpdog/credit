import logging
import numpy as np 
import lightgbm as lgb
from itertools import product
from sklearn.metrics import f1_score 


class LGB_GridSearchCV:
    def __init__(self, params, params_grid, nfold=3, num_boost_round=2000, early_stop=3,  verbose_eval=10):
        self.params = params
        self.params_grid = params_grid
        self.nfold = nfold
        self.num_boost_round = num_boost_round
        self.early_stop = early_stop
        self.verbose_eval = verbose_eval
        self.best_score = None
        self.best_params = {}

    def fit(self, lgb_data):
        print('\n')
        print('==================================================')
        print("開始搜索流程.")
        print("搜索參數 : ")
        print(self.params_grid)

        # ** 初始化最佳分數最佳數
        best_score = float("-inf")
        best_params_vec = None

        # ** 取得所有參數組合
        param_items = sorted(self.params_grid.items())
        param_keys, param_values = zip(*param_items)
        all_param_combinations = product(*param_values)
        n_params = len(param_keys)

        # **　遍歷所有參數組合，若分數提高則更新最佳分數及最佳參數
        for param_combination in all_param_combinations:
            print("\n測試參數組合 : ")
            for i in range(n_params):
                print("{} = {}".format(param_keys[i], param_combination[i]))
                self.params[param_keys[i]] = param_combination[i]

            # ** 取得交叉驗證分數
            cv_results = lgb.cv(self.params, lgb_data, num_boost_round=self.num_boost_round, 
                                nfold=self.nfold, early_stopping_rounds = self.early_stop,
                                verbose_eval=self.verbose_eval, stratified=False, feval=lgb_f1_score)
        
            # ** 更新最佳參數及分數
            current_score = np.array(cv_results.get("f1_score-mean")).max()
            boost_rounds = np.array(cv_results.get("f1_score-mean")).argmax()
            print('f1 score : {}'.format(current_score))
            print('num rounds : {}'.format(boost_rounds))

            if current_score > best_score:
                best_score = current_score
                best_params_vec = param_combination
                best_boost_rounds = boost_rounds

        # ** 結束搜索，記錄最佳分數及參數組
        self.best_score = best_score
        self.best_boost_rounds = best_boost_rounds
        print('\n')
        print('==================================================')
        print('搜索完成')
        print('最佳分數為 : ',format(self.best_score))
        print('最佳參數為 : ')
        for i in range(n_params):
            print("{} = {}".format(param_keys[i], best_params_vec[i]))
            self.best_params[param_keys[i]] = param_combination[i]

        return self

    def get_best_params(self):
        return self.best_params

    def get_best_score(self):
        return self.best_score

    def get_boost_rounds(self):
        return self.best_boost_rounds

def lgb_f1_score(preds, dtrain, threshold=0.5):
    """ lightgbm custom evaluate function """
    labels = dtrain.get_label()
    predition = 1 * (preds > threshold)
    score = f1_score(labels, predition)
    return 'f1_score', score, True