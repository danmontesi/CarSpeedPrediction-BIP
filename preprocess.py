import pandas as pd

from datetime import timedelta

from tqdm import tqdm


def preprocess():
    
    def compute_train(grouped, speeds_df):

        for name, group in tqdm(grouped):
            speeds_relevant = speeds_df[speeds_df.KEY == name]

            merged = pd.merge(group, speeds_relevant, on=['KEY'])

            merged = merged[(merged.DATETIME_UTC >= merged['START_DATETIME_UTC'] - timedelta(minutes=8)) & (
                    merged.DATETIME_UTC <= merged['START_DATETIME_UTC'] + timedelta(hours=1)) & (
                                    merged.KM >= merged['KM_START'] - 5) & (merged.KM <= merged['KM_END'] + 5)]

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

    grouped_events = events_df.groupby('KEY')

    compute_train(grouped_events, speeds_df)


if __name__ == "__main__":
    preprocess()
