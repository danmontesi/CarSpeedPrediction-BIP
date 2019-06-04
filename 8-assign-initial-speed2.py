import pandas as pd

from datetime import timedelta, datetime, time
#import datetime

from tqdm import tqdm

CLUSTER_SIZE = 20000
DATASET = "bip_assignment/dataset.csv"



def preprocess():



    dataset = pd.read_csv("bip_assignment/dataset_6.csv", dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object, "STATION_ID_4": object})
    #dataset_orig = pd.read_csv("bip_assignment/speeds_train.csv")

    #dataset_orig['DATETIME_UTC'] = pd.to_datetime(dataset_orig['DATETIME_UTC'])

    dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])
    dataset['START_DATETIME_UTC'] = pd.to_datetime(dataset['START_DATETIME_UTC'])
    dataset['END_DATETIME_UTC'] = pd.to_datetime(dataset['END_DATETIME_UTC'])

    print("##########################################################+")
    grouped_dataset = dataset[(dataset['READ_INSTANT'] != 1) & (dataset['READ_INSTANT'] != -9999)].groupby('KEY')

    for name, group in tqdm(grouped_dataset):
        reduced_orig = dataset[dataset['KEY'] == name]

        for i in tqdm(group.index):
            interesting = reduced_orig[(reduced_orig['KM'] == group.at[i, 'KM']) & (
                        reduced_orig['DATETIME_UTC'] == group.at[i, 'DATETIME_UTC'] - timedelta(minutes=15))]
            # print(interesting.head())
            if interesting.shape[0] > 0:
                # print(interesting.SPEED_AVG)
                dataset.loc[i, 'PREC_SPEED'] = float(interesting.SPEED_AVG.ravel()[0])
                # print(interesting.SPEED_AVG.ravel()[0])



    dataset.to_csv("bip_assignment/dataset_7.csv", index=False)


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)

    preprocess()
