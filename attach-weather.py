import pandas as pd

from datetime import timedelta
import numpy as np

from tqdm import tqdm

CLUSTER_SIZE = 20000
DATASET = "bip_assignment/dataset_with_all_km.csv"


def preprocess():
    def compute_weather(grouped, weather_df):


        for name, group in tqdm(grouped):
            #print(name)

            weather_relevant = weather_df[weather_df.ID == name]

            for i in (group.index):
                #line=group[group.index == i]

                weather_very_relevant = weather_relevant[(group.at[i,'DATETIME_UTC'] >= weather_relevant.DATETIME_UTC - timedelta(hours=12))
                                                         &  (group.at[i,'DATETIME_UTC'] <= weather_relevant.DATETIME_UTC + timedelta(hours=12))]
                weather_very_relevant["TIME_WEATHER_DELTA"] = abs(weather_relevant.DATETIME_UTC - group.at[i,'DATETIME_UTC'])
                #print(weather_very_relevant.head())
                index = int(weather_very_relevant[["TIME_WEATHER_DELTA"]].idxmin())
                #print(index)
                dataset.loc[i, 'TEMPERATURE'] = weather_very_relevant.at[index, "TEMPERATURE"]
                dataset.loc[i, 'MAX_TEMPERATURE'] = weather_very_relevant.at[index, "MAX_TEMPERATURE"]
                dataset.loc[i, 'MIN_TEMPERATURE'] = weather_very_relevant.at[index, "MIN_TEMPERATURE"]
                dataset.loc[i, 'WEATHER'] = weather_very_relevant.at[index, "WEATHER"]
                dataset.loc[i, 'DATETIME_UTC_WEATHER'] = weather_very_relevant.at[index, "DATETIME_UTC"]






    dataset = pd.read_csv("bip_assignment/dataset_2.csv")

    weather_df = pd.read_csv("bip_assignment/weather_train.csv")

    dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])
    dataset['START_DATETIME_UTC'] = pd.to_datetime(dataset['START_DATETIME_UTC'])
    dataset['END_DATETIME_UTC'] = pd.to_datetime(dataset['END_DATETIME_UTC'])

    dataset['TEMPERATURE'] = 0
    dataset['MAX_TEMPERATURE'] = 0
    dataset['MIN_TEMPERATURE'] = 0
    dataset['WEATHER'] = 0
    dataset['DATETIME_UTC_WEATHER'] = 0


    weather_df['DATETIME_UTC'] = pd.to_datetime(weather_df['DATETIME_UTC'])
    grouped_dataset = dataset.groupby('STATION_ID')

    print(dataset[dataset['STATION_ID'] == np.nan].head())

    compute_weather(grouped_dataset, weather_df)




    print("Fixing missing values first try...")
    missing_values = dataset[dataset.isnull().WEATHER == True]
    print(missing_values.shape[0])
    grouped_dataset = missing_values.groupby('STATION_ID_2')
    compute_weather(grouped_dataset, weather_df)

    print("Fixing missing values second try...")
    missing_values = dataset[dataset.isnull().WEATHER == True]
    print(missing_values.shape[0])
    grouped_dataset = missing_values.groupby('STATION_ID_3')
    compute_weather(grouped_dataset, weather_df)

    print("Fixing missing values third try...")
    missing_values = dataset[dataset.isnull().WEATHER == True]
    print(missing_values.shape[0])
    grouped_dataset = missing_values.groupby('STATION_ID_4')
    compute_weather(grouped_dataset, weather_df)

    print("Final missing:")
    missing_values = dataset[dataset.isnull().WEATHER == True]
    print(missing_values.shape[0])

    dataset.to_csv("./bip_assignment/dataset_3.csv")


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)
    preprocess()
