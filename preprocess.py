import numpy
import pandas as pd
import datetime

import pandas as pd

from datetime import timedelta

from tqdm import tqdm


def preprocess():
    def compute_train(grouped_speeds, events_df):

        for name, group in tqdm(grouped_speeds):
            events_relevant = events_df[events_df.KEY == name]

            merged = pd.merge(group, events_relevant, on=['KEY'])

            # Filtering

            merged = merged[(merged['START_DATETIME_UTC'] <= merged.DATETIME_UTC + timedelta(minutes=8)) & (
                    merged['START_DATETIME_UTC'] >= merged.DATETIME_UTC + timedelta(hours=1)) & (
                                    merged['KM_START'] <= merged.KM + 5) & (merged['KM_END'] >= merged.KM + 5)]

            print(merged.head(10))
            if name == 0:
                merged.to_csv(
                    'bip_assignment/dataset.csv')
            else:
                with open('bip_assignment/dataset.csv', 'a') as f:
                    merged.to_csv(f, header=False)

    speeds_df = pd.read_csv("bip_assignment/speeds_train.csv")

    events_df = pd.read_csv("bip_assignment/events_train.csv")

    speeds_df['DATETIME_UTC'] = pd.to_datetime(speeds_df['DATETIME_UTC'])
    events_df['START_DATETIME_UTC'] = pd.to_datetime(events_df['START_DATETIME_UTC'])
    events_df['END_DATETIME_UTC'] = pd.to_datetime(events_df['END_DATETIME_UTC'])

    grouped_speeds = speeds_df.groupby('KEY')

    compute_train(grouped_speeds, events_df)


if __name__ == "__main__":
    preprocess()
