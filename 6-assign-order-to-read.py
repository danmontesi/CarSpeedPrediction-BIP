import pandas as pd

from datetime import timedelta, datetime, time
#import datetime

from tqdm import tqdm

CLUSTER_SIZE = 20000
DATASET = "bip_assignment/dataset.csv"

def ceil_dt(dt, delta):
    print(dt)
    print(delta)
    print((time.min - dt))
    return dt + ((datetime.min - dt) % delta)


def preprocess():

    def roundData(x):
        if x<timedelta(minutes=15):
            return 1
        if x<timedelta(minutes=30):
            return 2
        if x<timedelta(minutes=45):
            return 3
        if x<timedelta(minutes=60):
            return 4
        return -9999

    dataset = pd.read_csv("bip_assignment/dataset_4.csv", dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object, "STATION_ID_4": object})
    dataset.dropna()
    dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])
    dataset['START_DATETIME_UTC'] = pd.to_datetime(dataset['START_DATETIME_UTC'])
    dataset['END_DATETIME_UTC'] = pd.to_datetime(dataset['END_DATETIME_UTC'])

    dataset['APPROX_TIME'] = dataset['START_DATETIME_UTC'].dt.ceil('15min')

    #dataset = approximate_time(dataset)

    dataset['DELTA_TIME'] = dataset['DATETIME_UTC'] - dataset['APPROX_TIME']
    dataset['DISTANCE_FROM_EVENT_MINUTES'] = (dataset['DATETIME_UTC'] - dataset['START_DATETIME_UTC']).apply(lambda x:  x.total_seconds()/60)

    dataset['READ_INSTANT'] = (dataset['DELTA_TIME']).apply(lambda x: roundData(x))



    dataset.to_csv("bip_assignment/dataset_5.csv", index=False)


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)

    preprocess()
