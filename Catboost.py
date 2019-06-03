import math
import time
from abc import ABC
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from catboost import CatBoost, Pool, CatBoostRegressor
from copy import deepcopy
import pickle
import pandas as pd

tqdm.pandas()


class CatboostRegressor(ABC):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """

    def __init__(self, train, test=None, learning_rate=0.5, iterations=20, max_depth=10, reg_lambda=6.0,
                 custom_metric='AverageGain:top=1', one_hot_max_size = 50, include_test = True,
                 features_to_one_hot=None):

        self.features_to_drop = []
        self.dataset_name = 'catboost_rank'

        self.train_df = train
        self.test_df = test

        self.include_test = include_test

        self.iterations = math.ceil(iterations)
        self.custom_metric = custom_metric
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.one_hot_max_size = one_hot_max_size

        self.features_to_one_hot = features_to_one_hot
        self.ctb = None
        self.categorical_features = None
        self.train_features = None
        self.scores_batch = None

    def fit_model(self, train_pool=None, test_pool=None):
        model = CatBoostRegressor(iterations=self.iterations, learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                  max_depth=self.max_depth, one_hot_max_size=self.one_hot_max_size, eval_metric='MAE')
        start = time.time()

        model.fit(train_pool, eval_set=test_pool, verbose=True)

        end = time.time()
        print('With {} iteration, training took {} sec'.format(self.iterations, end - start))

        return model

    def fit(self):
        train_df = self.train_df
        train_df = train_df.fillna(0)

        # PREPROCESS: prepare for train
        train_features = train_df

        features = list(train_features.drop('SPEED_AVG', axis=1).columns.values)
        self.categorical_features = []
        for f in features:
            if isinstance(train_features.head(1)[f].values[0], str):
                print(train_features.head(1)[f].values[0])
                self.categorical_features.append(features.index(f))
                print(f + ' is categorical!')

        if len(self.categorical_features) == 0:
            self.categorical_features = None

        self.train_features = train_features.columns.values
        X_train = train_features.drop('SPEED_AVG', axis=1).values
        y_train = train_df['SPEED_AVG'].values

        # Creating pool for training data
        train_with_weights = Pool(
            data=X_train,
            label=y_train,
            cat_features=self.categorical_features
        )

        test_with_weights = None

        if self.include_test:
            test_df = self.test_df
            test_df = test_df.fillna(0)

            if list(test_df.columns.values) == list(train_df.columns.values):
                print('NOT SAME SHAPE of train and test, fix it up')

            X_test = test_df.drop(['SPEED_AVG'], axis=1).values
            y_test = test_df['SPEED_AVG'].values

            print("pooling")

            test_with_weights = Pool(
                data=X_test,
                label=y_test,
                cat_features=self.categorical_features
            )

        print('data for train ready')

        self.ctb = self.fit_model(train_pool=train_with_weights,
                                  test_pool=test_with_weights)
        print('fit done')


    def recommend_batch(self):
        test_df = self.test_df

        X_test = test_df.drop(['SPEED_AVG'], axis=1).values
        y_test = test_df['SPEED_AVG'].values

        print("pooling")

        test_with_weights = Pool(
            data=X_test,
            label=None,
            cat_features=self.categorical_features
        )



if __name__ == '__main__':
    model = CatboostRegressor(pd.read_csv('final_dataset/train.csv'), pd.read_csv('final_dataset/validation.csv'))
    model.fit()



