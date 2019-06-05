import pandas as pd

from datetime import timedelta

from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


def preprocess():
    speeds_df = pd.read_csv("output.csv",
                            dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object,
                                   "STATION_ID_4": object})
    print(speeds_df.shape[0])

    speeds_df['DATETIME_UTC'] = pd.to_datetime(speeds_df['DATETIME_UTC'])
    orig_df = pd.read_csv("../bip_assignment/test/speeds_test.csv")
    orig_df['DATETIME_UTC'] = pd.to_datetime(orig_df['DATETIME_UTC'])



    #final=pd.DataFrame(columns=['KEY', 'KM', 'DATETIME_UTC','PREDICTION_STEP', 'SPEED_AVG'])
    grouped = speeds_df.groupby(['KEY', 'KM', 'DATETIME_UTC', 'READ_INSTANT'])

    print(grouped.mean())
    #FIXME: DOES NOT WORK!!!

    grouped=grouped.merge(orig_df[['KEY', 'KM', 'DATETIME_UTC', 'SPEED_AVG']], on=['KEY', 'KM', 'DATETIME_UTC'], how='inner', )
    print(grouped.head(25))
    print(grouped.shape[0])
    print(mean_absolute_error(grouped['PREDICTION'], grouped['SPEED_AVG']))


if __name__ == "__main__":
    preprocess()
