from multiprocessing.pool import Pool

import pandas as pd

from datetime import timedelta
import numpy as np

from tqdm import tqdm

CLUSTER_SIZE = 20000
DATASET = "bip_assignment/dataset.csv"



def compute_weather(key):
    dataset = pd.read_csv("bip_assignment/dataset_stations.csv")

    weather_df = pd.read_csv("bip_assignment/weather_train.csv")

    dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])
    dataset['START_DATETIME_UTC'] = pd.to_datetime(dataset['START_DATETIME_UTC'])
    dataset['END_DATETIME_UTC'] = pd.to_datetime(dataset['END_DATETIME_UTC'])

    weather_df['DATETIME_UTC'] = pd.to_datetime(weather_df['DATETIME_UTC'])


    print(key)

    group = dataset[dataset.STATION_ID_2 == key]

    weather_relevant = weather_df[weather_df.ID == key]

    for i in tqdm(group.index):
        #line=group[group.index == i]

        weather_very_relevant = weather_relevant[(group.at[i,'DATETIME_UTC'] >= weather_relevant.DATETIME_UTC - timedelta(hours=12))
                                                 &  (group.at[i,'DATETIME_UTC'] <= weather_relevant.DATETIME_UTC + timedelta(hours=12))]
        weather_very_relevant["TIME_WEATHER_DELTA"] = abs(weather_relevant.DATETIME_UTC - group.at[i,'DATETIME_UTC'])
        #print(weather_very_relevant.head())
        index = int(weather_very_relevant[["TIME_WEATHER_DELTA"]].idxmin())
        #print(index)
        group.loc[i, 'TEMPERATURE'] = weather_very_relevant.at[index, "TEMPERATURE"]
        group.loc[i, 'MAX_TEMPERATURE'] = weather_very_relevant.at[index, "MAX_TEMPERATURE"]
        group.loc[i, 'MIN_TEMPERATURE'] = weather_very_relevant.at[index, "MIN_TEMPERATURE"]
        group.loc[i, 'WEATHER'] = weather_very_relevant.at[index, "WEATHER"]
        group.loc[i, 'DATETIME_UTC_WEATHER'] = weather_very_relevant.at[index, "DATETIME_UTC"]

    group.to_csv("dataset_station1_"+key+".csv")





dataset_total = pd.read_csv("bip_assignment/dataset_2.csv")

stations = set(dataset_total["STATION_ID_2"])

print(len(set(dataset_total["STATION_ID_2"])))

p = Pool(8)
p.map(compute_weather, tqdm(stations))

