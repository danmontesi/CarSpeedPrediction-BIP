import math
import time
from abc import ABC
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from copy import deepcopy
import pickle
import pandas as pd
from xgboost import XGBRegressor

tqdm.pandas()


class XGBoostRegressor(ABC):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    """

    def __init__(self, train, test=None, learning_rate=0.5, iterations=20, max_depth=10, reg_lambda=6.0,
                 custom_metric='AverageGain:top=1', reg_alpha = 5.0, include_test = True,
                 cat_features = []):

        self.reg_alpha = reg_alpha
        self.features_to_drop = []
        self.dataset_name = 'catboost_rank'
        self.cat_features = cat_features

        self.train_df = train
        self.test_df = test

        self.include_test = include_test

        self.n_estimators = math.ceil(iterations)
        self.custom_metric = custom_metric
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda

        self.ctb = None
        self.categorical_features = None
        self.train_features = None
        self.scores_batch = None

    def fit_model(self, X, y, X_test=None, y_test=None):
        model = XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                             max_depth=self.max_depth, reg_alpha=self.reg_alpha, objective='reg:gamma')
        start = time.time()

        if X_test is not None:
            eval_set = (X_test, y_test)
        else:
            eval_set = None
        model.fit(X, y, eval_set= eval_set, eval_metric='mae')

        end = time.time()
        print('With {} iteration, training took {} sec'.format(self.n_estimators, end - start))

        return model

    def fit(self):
        train_df = self.train_df
        train_df = train_df.fillna(0)

        # PREPROCESS: prepare for train
        train_features = train_df

        self.train_features = train_features.columns.values
        X_train = train_features.drop('SPEED_AVG', axis=1).values
        y_train = train_df['SPEED_AVG'].values

        print('data for train ready')

        if self.include_test:
            test_df = self.test_df
            test_df = self.preprocess_dataset(test_df)
            test_df = test_df.fillna(0)

            if list(test_df.columns.values) != list(train_df.columns.values):
                print('NOT SAME SHAPE of train and test, fix it up')

            X_test = test_df.drop(['SPEED_AVG'], axis=1).values
            y_test = test_df['SPEED_AVG'].values

            self.ctb = self.fit_model(X_train, y_train, X_test, y_test)
        else:
            self.ctb = self.fit_model(X_train, y_train)

        print('fit done')

    def preprocess_dataset(self, dataset):

        #Create onehot features
        # one-hot encoding delle var categoriche
        dataset = dataset.drop(['APPROX_TIME', 'DATETIME_UTC'], axis=1)

        for i in tqdm(self.cat_features):
            one_hot = pd.get_dummies(dataset[i])
            dataset = dataset.drop(i, axis=1)
            dataset = dataset.join(one_hot)

        return dataset


if __name__ == '__main__':
    model = XGBoostRegressor(pd.read_csv('final_dataset/train.csv'), pd.read_csv('final_dataset/validation.csv'), cat_features=['EVENT_DETAIL', 'EVENT_TYPE', 'WEEK_DAY', 'TIME_INTERVAL', 'ROAD_TYPE', 'DELTA_TIME', 'WEATHER'])
    model.fit()



