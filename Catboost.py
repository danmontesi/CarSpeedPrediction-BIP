import math
import time
from abc import ABC
from datetime import timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from catboost import CatBoost, Pool, CatBoostRegressor
from copy import deepcopy
import pickle
import pandas as pd
import random
tqdm.pandas()


class CatboostRegressor(ABC):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """

    def __init__(self, train, test=None, cat_features=None, learning_rate=0.1, iterations=320, max_depth=9, reg_lambda=6.0,
                 custom_metric='AverageGain:top=1', one_hot_max_size = 30, include_test = True,
                 features_to_one_hot=None):

        self.features_to_drop = []
        self.dataset_name = 'catboost_rank'

        self.train_df = train
        self.test_df = test

        self.include_test = include_test
        self.test_with_weights=None

        self.iterations = math.ceil(iterations)
        self.custom_metric = custom_metric
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.one_hot_max_size = one_hot_max_size

        self.features_to_one_hot = features_to_one_hot
        self.ctb = None
        self.cat_features = cat_features
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

        if self.cat_features is not None:
            self.categorical_features = []

            for f in self.cat_features:
                    self.categorical_features.append(features.index(f))

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
            test_df = self.test_df.drop(['DATETIME_UTC', 'KM', 'KEY', 'DELTA_TIME'], axis=1)
            test_df = test_df.fillna(0)

            if list(test_df.columns.values) != list(train_df.columns.values):
                print('NOT SAME SHAPE of train and test, fix it up')
                print(test_df.columns.values)
                print(train_df.columns.values)

            X_test = test_df.drop(['SPEED_AVG'], axis=1).values
            y_test = test_df['SPEED_AVG'].values

            print("pooling")

            self.test_with_weights = Pool(
                data=X_test,
                label=y_test,
                cat_features=self.categorical_features
            )

        print('data for train ready')

        self.ctb = self.fit_model(train_pool=train_with_weights,
                                  test_pool=self.test_with_weights)
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

    def predict(self, topredict):

        if self.test_with_weights is not None:
            out = self.ctb.predict(self.test_with_weights, verbose=True)
        else:
            if 'SPEED_AVG' in topredict.columns.values:
                topredict.drop('SPEED_AVG', axis=1, inplace=True)
            X_test = topredict.drop(['DATETIME_UTC', 'KM', 'KEY', 'DELTA_TIME'], axis=1).fillna(0).values

            out = self.ctb.predict(X_test, verbose = True)
        print(out)
        return out



def catchLastSpeed(dataset, iter):
    grouped_dataset = dataset[(dataset['READ_INSTANT'] == iter)].groupby('KEY')

    for name, group in tqdm(grouped_dataset):
        reduced_orig = dataset[dataset['KEY'] == name]

        for i in (group.index):
            interesting = reduced_orig[(reduced_orig['KM'] == group.at[i, 'KM']) & (
                    reduced_orig['DATETIME_UTC'] == group.at[i, 'DATETIME_UTC'] - timedelta(minutes=15))]
            # print(interesting.head())
            if interesting.shape[0] > 0:
                # print(interesting.SPEED_AVG)
                dataset.loc[i, 'PREC_SPEED'] = float(interesting.PREDICTION.ravel()[0])
                # print(interesting.SPEED_AVG.ravel()[0])


def train_read_instant(train, validation_data, iter, catch=True):



    model = CatboostRegressor(train,
                              validation_data[validation_data.READ_INSTANT == iter].drop('PREDICTION', axis=1),
                              include_test=True,
                              cat_features=['EVENT_DETAIL', 'EVENT_TYPE', 'WEEK_DAY', 'TIME_INTERVAL', 'ROAD_TYPE', 'WEATHER' ])
    model.fit()


    out=model.predict(validation_data[validation_data.READ_INSTANT==iter])
    validation_data.at[validation_data.READ_INSTANT==iter, "PREDICTION"] = out
    print(validation_data.head(25))
    if catch:
        catchLastSpeed(validation_data, iter+1)

    return validation_data


def old_train():
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)


    validation_data = pd.read_csv('preprocess-test/finalvtest.csv').drop(['APPROX_TIME'], axis=1)
    validation_data['DATETIME_UTC'] = pd.to_datetime(validation_data['DATETIME_UTC'])
    #validation_data['START_DATETIME_UTC'] = pd.to_datetime(validation_data['START_DATETIME_UTC'])
    #validation_data['END_DATETIME_UTC'] = pd.to_datetime(validation_data['END_DATETIME_UTC'])

    #validation_train_data = pd.read_csv('final_dataset/test.csv').drop(['APPROX_TIME'], axis=1)


    model = CatboostRegressor(pd.read_csv('final_dataset/merged.csv', dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object, "STATION_ID_4": object}).drop(['Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'APPROX_TIME', 'DATETIME_UTC', 'KM', 'KEY', 'DELTA_TIME'], axis=1),
                              #validation_train_data.drop(['Unnamed: 0.1.1', 'Unnamed: 0.1.1.1','DATETIME_UTC', 'KM', 'KEY', 'DELTA_TIME'], axis=1),
                              include_test=False,
                              cat_features=['EVENT_DETAIL', 'EVENT_TYPE', 'WEEK_DAY', 'TIME_INTERVAL', 'ROAD_TYPE', 'WEATHER', 'READ_INSTANT'])
    model.fit()

    validation_data["PREDICTION"] = float(0)

    out=model.predict(validation_data[validation_data.READ_INSTANT == 1])
    validation_data.at[validation_data.READ_INSTANT==1, "PREDICTION"] = out
    print(validation_data.head(25))

    catchLastSpeed(validation_data, 2)
    out=model.predict(validation_data[validation_data.READ_INSTANT == 2])
    validation_data.at[validation_data.READ_INSTANT == 2, "PREDICTION"] = out
    print(validation_data.head(25))

    catchLastSpeed(validation_data, 3)
    out=model.predict(validation_data[validation_data.READ_INSTANT == 3])
    validation_data.at[validation_data.READ_INSTANT == 3, "PREDICTION"] = out
    print(validation_data.head(25))

    catchLastSpeed(validation_data, 4)
    out=model.predict(validation_data[validation_data.READ_INSTANT == 4])
    validation_data.at[validation_data.READ_INSTANT == 4, "PREDICTION"] = out
    print(validation_data.head(25))


    validation_data.to_csv("preprocess-test/output.csv")
    print(validation_data.head(25))

def validate(train, validation_data):

    validation_data = validation_data[validation_data.READ_INSTANT == 1]
    while True:
        one_hot_max_size = random.randint(6, 50)
        max_depth = random.randint(3,12)
        reg_lambda = random.uniform(1, 8)

        print('Fitting with onehot_max: {}, max_depth: {}, reg_lambda: {}'.format(one_hot_max_size, max_depth, reg_lambda))
        model = CatboostRegressor(train, validation_data,
                                  include_test=True,
                                  cat_features=['EVENT_DETAIL', 'EVENT_TYPE', 'WEEK_DAY', 'TIME_INTERVAL', 'ROAD_TYPE',
                                                'WEATHER'],
                                  one_hot_max_size=one_hot_max_size, max_depth=max_depth, reg_lambda=reg_lambda, learning_rate=0.3, iterations=100)
        model.fit()
        print('onehot_max: {}, max_depth: {}, reg_lambda: {}'.format(one_hot_max_size, max_depth,
                                                                                  reg_lambda))


def train_separate_read_instant():
    validation_data = pd.read_csv('preprocess-test/test_final.csv').drop(['APPROX_TIME'], axis=1)
    validation_data['DATETIME_UTC'] = pd.to_datetime(validation_data['DATETIME_UTC'])
    # validation_data['START_DATETIME_UTC'] = pd.to_datetime(validation_data['START_DATETIME_UTC'])
    # validation_data['END_DATETIME_UTC'] = pd.to_datetime(validation_data['END_DATETIME_UTC'])

    # validation_train_data = pd.read_csv('final_dataset/test.csv').drop(['APPROX_TIME'], axis=1)

    train = pd.read_csv('final_dataset/merged.csv',
                        dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object,
                               "STATION_ID_4": object}).drop(
        ['Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'APPROX_TIME', 'KM', 'KEY', 'DATETIME_UTC', 'DELTA_TIME'], axis=1)

    # train_correct = train[train.READ_INSTANT == 2]
    # # Adding bias to remove dependancy from prec value
    # for i in tqdm(train_correct.index):
    #     train.at[i, 'PREC_SPEED'] = train.at[i, 'PREC_SPEED'] + (6 / random.randint(1, 3)) * random.randint(-3, 3)
    #
    # train_correct = train[train.READ_INSTANT == 3]
    # # Adding bias to remove dependancy from prec value
    # for i in tqdm(train_correct.index):
    #     train.at[i, 'PREC_SPEED'] = train.at[i, 'PREC_SPEED'] + (7 / random.randint(1, 3)) * random.randint(-3, 3)
    #
    # train_correct = train[train.READ_INSTANT == 4]
    # # Adding bias to remove dependancy from prec value
    # for i in tqdm(train_correct.index):
    #     train.at[i, 'PREC_SPEED'] = train.at[i, 'PREC_SPEED'] + (8 / random.randint(1, 3)) * random.randint(-3, 3)

    validation_data["PREDICTION"] = float(0)

    validation_data = train_read_instant(train, validation_data, 1, catch=True)

    validation_data = train_read_instant(train, validation_data, 2, catch=True)

    validation_data = train_read_instant(train, validation_data, 3, catch=True)

    validation_data = train_read_instant(train, validation_data, 4, catch=False)

    # validation_data['ERROR'] = abs(validation_data['PREDICTION'] - validation_data['SPEED_AVG'])

    # validation_data['ERROR_PER'] = validation_data['ERROR']/validation_data['SPEED_AVG'] * 100

    out_full = validation_data.PREDICTION

    validation_data.to_csv("preprocess-test/output.csv")

    print(validation_data.head(25))
    # print(validation_data[validation_data.READ_INSTANT == 4]['ERROR'].describe())
    # print(validation_data[validation_data.READ_INSTANT == 4]['ERROR_PER'].describe())


if __name__ == '__main__':
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)


    #train_separate_read_instant()
    #train_once()

    train = pd.read_csv('final_dataset/merged.csv',
                        dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object,
                               "STATION_ID_4": object}).drop(
        ['Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'APPROX_TIME', 'KM', 'KEY', 'DATETIME_UTC', 'DELTA_TIME'], axis=1)
    validation_data = pd.read_csv('preprocess-test/test_final.csv').drop(['APPROX_TIME'], axis=1)
    validation_data['DATETIME_UTC'] = pd.to_datetime(validation_data['DATETIME_UTC'])

    validate(train, validation_data)
