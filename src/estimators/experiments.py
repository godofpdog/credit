import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
 

class Estimator:
    def __init__(self):
        pass 

    def fit(self, train_data, target_colname):
        """ 訓練模型 """

        # ** 轉換資料
        X = train_data.drop(columns=target_colname).values
        y = train_data[target_colname].values

        dtrain = lgb.Dataset(X, label=y, categorical_feature=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20])
        
        # ** lgb參數
        params = {'max_depth':-1, 
                  'learning_rate':0.1,
                  'num_leaves':511,
                  'feature_fraction':0.7,
                  'min_split_gain':0,
                  'bagging_freq':1,
                  'objective':'binary', 
                  'num_threads':4, 
                  'boost_from_average':False,
                  'min_data_in_leaf':10,
                  'is_unbalance':True,
                  'verbose':-1
                  }
        # ** CV
        # print('\n** 內部驗證')
        self.cv_results = lgb.cv(params, dtrain, 
                                 num_boost_round=10,
                                 nfold=5, 
                                 early_stopping_rounds=10,
                                 metrics=['evalerror'], 
                                 feval=evalerror,
                                 verbose_eval=100, 
                                 stratified=False)
        
        cv_score = np.array(self.cv_results.get("f1_score-mean")).max()
        num_round = np.array(self.cv_results.get("f1_score-mean")).argmax()

        # print('f1_score-mean :', cv_score)
        # print('nun round : ', num_round)

        # ** re-train
        dtrain = lgb.Dataset(X, label=y, categorical_feature=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20])
        self.bst = lgb.train(params, dtrain, num_round, valid_sets=[dtrain], feval=evalerror, verbose_eval=100)

    def evaluate(self, test_data, target_colname):
        """ 回傳驗證評估分數 """

        # ** 轉換資料
        X = test_data.drop(columns=target_colname).values
        y = test_data[target_colname].values

        threshold = .5
        bst_predictions = 1 * (self.bst.predict(X) > threshold)
        self.score = f1_score(y, bst_predictions)
        return self.score

    def infer(self, input_data):
        """  回傳推論結果 """
        X = input_data.values
        threshold = 0.5
        return 1 * (self.bst.predict(X) > threshold)

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    predition = 1 * (preds>0.5)
    f_score = f1_score(labels, predition)
    return 'f1_score', f_score, True


