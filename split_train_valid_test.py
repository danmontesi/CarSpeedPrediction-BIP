
import datetime

import pandas as pd

def split(dataset):
    dataset = dataset.sort_values(by=['DATETIME_UTC'])

    thresh_train = datetime.datetime(2018, 11, 7)
    thresh_test = datetime.datetime(2018, 11, 23)

    col_to_drop = 'Unnamed: 0,Unnamed: 0.1,START_DATETIME_UTC,END_DATETIME_UTC,KM_END,KM_START,KEY_2_x,SPEED_SD,SPEED_MIN,SPEED_MAX,N_VEHICLES,KEY_2_y,STATION_ID,STATION_ID_2,STATION_ID_3,STATION_ID_4,DATETIME_UTC_WEATHER'.split(
        ',')
    dataset.drop(col_to_drop, axis=1, inplace=True)

    train = dataset[dataset.DATETIME_UTC <= thresh_train]

    validation = dataset[(dataset.DATETIME_UTC > thresh_train)&(dataset.DATETIME_UTC <= thresh_test)]

    test = dataset[(dataset.DATETIME_UTC > thresh_test)]

    train.to_csv('final_dataset/train.csv', index=False)

    validation.to_csv('final_dataset/validation.csv', index=False)

    test.to_csv('final_dataset/test.csv', index=False)

    print('Done!')

dataset = pd.read_csv("bip_assignment/dataset_4.csv", dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object, "STATION_ID_4": object})
dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])
split(dataset)
