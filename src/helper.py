#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:33:23 2018
helper.py
@author: liuyi
"""

import xgboost as xgb
from sklearn.model_selection import KFold
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class stacking(object):
    
    def __init__(self, X_train, y_train, X_test, n_folds, random_state):
        self.n_tr    = X_train.shape[0]
        self.n_te    = X_test.shape[0]      
        self.X_tr    = X_train
        self.y_tr    = y_train
        self.X_te    = X_test
        self.n_folds = n_folds
        self.kf      = KFold(n_splits = self.n_folds, random_state = random_state)
        self.estimators_dict  = {}
        self.n_features       = None
        self.meta_estimator   = None
        self.meta_features_tr = None
        self.meta_features_te = None
        

    def get_oof(self, estimator):
        oof_train    = np.zeros((self.n_tr,))
        oof_test     = np.zeros((self.n_te,))
        oof_test_kf  = np.zeros((self.n_folds, self.n_te))
        for i, (train_idx, test_idx) in enumerate(self.kf.split(self.X_tr)):
            print("fit fold {}.....".format(i + 1))
            X_tr_folds = self.X_tr[train_idx]
            X_te_folds = self.X_tr[test_idx]
            y_tr_folds = self.y_tr[train_idx]
            estimator.fit(X_tr_folds, y_tr_folds)
            oof_train[test_idx] = estimator.predict(X_te_folds)
            oof_test_kf[i, :]   = estimator.predict(self.X_te)
        oof_test[:] = oof_test_kf.mean(axis = 0)
        print("done.")
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    def get_oof_tocsr(self, estimator):
        oof_train    = np.zeros((self.n_tr,))
        oof_test     = np.zeros((self.n_te,))
        oof_test_kf  = np.zeros((self.n_folds, self.n_te))
        for i, (train_idx, test_idx) in enumerate(self.kf.split(self.X_tr)):
            print("fit fold {}.....".format(i + 1))
            X_tr_folds = self.X_tr.tocsr()[train_idx]
            X_te_folds = self.X_tr.tocsr()[test_idx]
            y_tr_folds = self.y_tr[train_idx]
            estimator.fit(X_tr_folds, y_tr_folds)
            oof_train[test_idx] = estimator.predict(X_te_folds)
            oof_test_kf[i, :]   = estimator.predict(self.X_te)
        oof_test[:] = oof_test_kf.mean(axis = 0)
        print("done.")
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    def get_all_oof(self, Sparse = False):
        self.n_features = len(self.estimators_dict.keys())
        self.meta_features_tr = np.zeros((self.n_tr, self.n_features))
        self.meta_features_te = np.zeros((self.n_te, self.n_features))
        i = 0
        for est_name in self.estimators_dict.keys():
            print("fit {} to get oof : ".format(est_name))
            if Sparse == True:
                meta_features_tr, meta_features_te = self.get_oof_tocsr(self.estimators_dict[est_name])
                self.meta_features_tr[:, i], self.meta_features_te[:, i] = \
                meta_features_tr.ravel(), meta_features_te.ravel()
            else:
                meta_features_tr, meta_features_te = self.get_oof(self.estimators_dict[est_name])
                self.meta_features_tr[:, i], self.meta_features_te[:, i] = \
                meta_features_tr.ravel(), meta_features_te.ravel()
            i += 1
        return None
    
    def add_base_learners(self, estimators):
        """ estimators : dict like with models """
        for est_name in estimators.keys():
            print("add base learner : {}.".format(est_name))
            self.estimators_dict[est_name] = estimators[est_name]
        return None
    
    def add_blr_get_oof(self, new_estimators, Sparse = False):
        """ new_estimators : dict like with models """
        """ add base learner  """
        self.add_base_learners(new_estimators)
        """ add meta features """
        tmp_n_features = len(new_estimators.keys())
        tmp_meta_features_tr = np.zeros((self.n_tr, tmp_n_features))
        tmp_meta_features_te = np.zeros((self.n_te, tmp_n_features))
        i = 0
        for est_name in new_estimators.keys():
            print("fit {} to get oof : ".format(est_name))
            if Sparse == True:
                meta_features_tr, meta_features_te = self.get_oof_tocsr(new_estimators[est_name])
                tmp_meta_features_tr[:, i], tmp_meta_features_te[:, i] = \
                meta_features_tr.ravel(), meta_features_te.ravel()
            else:
                meta_features_tr, meta_features_te = self.get_oof(new_estimators[est_name])
                tmp_meta_features_tr[:, i], tmp_meta_features_te[:, i] = \
                meta_features_tr.ravel(), meta_features_te.ravel()
            i += 1       
            
        if self.meta_features_tr == None:
            self.meta_features_tr = tmp_meta_features_tr
            self.meta_features_te = tmp_meta_features_te
        else:
            print("combine existed meta features.....")
            self.meta_features_tr = np.hstack((self.meta_features_tr, tmp_meta_features_tr))
            self.meta_features_te = np.hstack((self.meta_features_te, tmp_meta_features_te))
            print("done.")
        return None
    
    def meta_learner(self, meta_estimator):
        pass
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from itertools import product
class lgbGridSearch_helper(object):
    
    def __init__(self, params, params_grid, n_estimators = 2000, early_stop = 10, verbose_eval = 50,
                  metrics = 'rmse'):
        self.params       = params
        self.n_estimators = n_estimators
        self.early_stop   = early_stop
        self.verbose_eval = verbose_eval
        self.params_grid  = params_grid
        self.metrics      = metrics
        self.best_score_  = None
        self.best_params_ = {}
        self.params[metrics] = metrics
        
    def fit(self, dtrain, dvalid):
        print("start Grid-search procedure.")
        print("will try : ")
        print(self.params_grid)
        min_metric = float("Inf")
        best_params_vec = None
        param_items = sorted(self.params_grid.items())
        param_keys, param_values = zip(*param_items)
        all_param_combinations = product(*param_values)
        n_params = len(param_keys)
        for param_combination in all_param_combinations:
            print("\ntrain model with : ")
            for i in range(n_params):
                print("{} = {}".format(param_keys[i], param_combination[i]))
                self.params[param_keys[i]] = param_combination[i]
            # train and validate
            evals_result = {}
            model = lgb.train(self.params, dtrain, valid_sets = [dtrain, dvalid], valid_names = ["train", "valid"],
                              num_boost_round = self.n_estimators,
                              early_stopping_rounds = self.early_stop, evals_result = evals_result,
                              verbose_eval = self.verbose_eval)
            # update metrics
            boost_rounds = model.current_iteration() - 1
            new_metric   = evals_result["valid"][self.metrics][boost_rounds]
            if new_metric < min_metric:
                min_metric = new_metric
                best_params_vec = param_combination
        print("\nGrid-search procedure complete.")
        print("The best params : ")
        self.best_score_ = min_metric      
        for i in range(n_params):
            print("{} = {}".format(param_keys[i], best_params_vec[i]))
            self.best_params_[param_keys[i]] = param_combination[i]
        print("with {} = {}".format(str(self.metrics), min_metric))    
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class lgbGridSearchCV_helper(object):
    
    def __init__(self, params, params_grid, n_estimators = 2000, cv = 3, early_stop = 10, verbose_eval = 50,
                  metrics = {'rmse'}, seed = 0):
        self.params       = params
        self.n_estimators = n_estimators
        self.cv           = cv
        self.early_stop   = early_stop
        self.verbose_eval = verbose_eval
        self.params_grid  = params_grid
        self.metrics      = metrics
        self.seed         = seed
        self.best_score_  = None
        self.best_params_ = {}
        
    def fit(self, X, y):
        print("start Grid-search CV procedure.")
        print("will try : ")
        print(self.params_grid)
        min_metric = float("Inf")
        best_params_vec = None
        param_items = sorted(self.params_grid.items())
        param_keys, param_values = zip(*param_items)
        all_param_combinations = product(*param_values)
        n_params = len(param_keys)
        for param_combination in all_param_combinations:
            print("\nCV with : ")
            for i in range(n_params):
                print("{} = {}".format(param_keys[i], param_combination[i]))
                self.params[param_keys[i]] = param_combination[i]
            # run cv
            dtrain = lgb.Dataset(X, label = y)
            cv_results = lgb.cv(self.params, dtrain, num_boost_round = self.n_estimators, seed = self.seed,
                               nfold = self.cv, metrics = self.metrics, early_stopping_rounds = self.early_stop,
                               verbose_eval = self.verbose_eval, stratified = False)
            # update metrics
            mean_metrics = np.array(cv_results.get("rmse-mean")).min()
            boost_rounds = np.array(cv_results.get("rmse-mean")).argmin()
            print("\trmse = {} for {} rounds".format(mean_metrics, boost_rounds))
            if mean_metrics < min_metric:
                min_metric = mean_metrics
                best_params_vec = param_combination
        print("\nGrid-search CV procedure complete.")
        print("The best params : ")
        self.best_score_ = min_metric      
        for i in range(n_params):
            print("{} = {}".format(param_keys[i], best_params_vec[i]))
            self.best_params_[param_keys[i]] = param_combination[i]
        print("with {} = {}".format(str(list(self.metrics)[0]), min_metric))
        
        
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""             
from itertools import product
class xgbGridSearchCV_helper(object):
    
    def __init__(self, params, params_grid, n_estimators = 2000, cv = 3, early_stop = 10, verbose_eval = 50,
                  metrics = {'rmse'}, seed = 0):
        self.params       = params
        self.n_estimators = n_estimators
        self.cv           = cv
        self.early_stop   = early_stop
        self.verbose_eval = verbose_eval
        self.params_grid  = params_grid
        self.metrics      = metrics
        self.seed         = seed
        self.best_score_  = None
        self.best_params_ = {}
        
    def fit(self, X, y):
        print("start Grid-search CV procedure.")
        print("will try : ")
        print(self.params_grid)
        min_metric = float("Inf")
        best_params_vec = None
        param_items = sorted(self.params_grid.items())
        param_keys, param_values = zip(*param_items)
        all_param_combinations = product(*param_values)
        n_params = len(param_keys)
        for param_combination in all_param_combinations:
            print("\nCV with : ")
            for i in range(n_params):
                print("{} = {}".format(param_keys[i], param_combination[i]))
                self.params[param_keys[i]] = param_combination[i]
            # run cv
            dtrain = xgb.DMatrix(X, label = y)
            cv_results = xgb.cv(self.params, dtrain, num_boost_round = self.n_estimators, seed = self.seed,
                               nfold = self.cv, metrics = self.metrics, early_stopping_rounds = self.early_stop,
                               verbose_eval = self.verbose_eval)
            # update metrics
            metrics_name = "test-" + str(list(self.metrics)[0]) + "-mean"
            mean_metrics = cv_results.get(metrics_name).min()
            boost_rounds = cv_results.get(metrics_name).values.argmin()
            print("\trmse = {} for {} rounds".format(mean_metrics, boost_rounds))
            if mean_metrics < min_metric:
                min_metric = mean_metrics
                best_params_vec = param_combination
        print("\nGrid-search CV procedure complete.")
        print("The best params : ")
        self.best_score_ = min_metric      
        for i in range(n_params):
            print("{} = {}".format(param_keys[i], best_params_vec[i]))
            self.best_params_[param_keys[i]] = best_params_vec[i]
        print("with {} = {}".format(str(list(self.metrics)[0]), min_metric))
        
        return best_params_vec
        
        
        

        