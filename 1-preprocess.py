import pandas as pd

from datetime import timedelta

from tqdm import tqdm

CLUSTER_SIZE = 20000
DATASET = "bip_assignment/dataset.csv"


def preprocess():
    def compute_train(grouped, speeds_df):
        processed = 0

        for name, group in tqdm(grouped):
            speeds_relevant = speeds_df[speeds_df.KEY == name]

            while speeds_relevant.shape[0] > CLUSTER_SIZE:
                merged = pd.merge(group, speeds_relevant[:CLUSTER_SIZE], on=['KEY'])

                merged = merged[(merged.DATETIME_UTC >= merged['START_DATETIME_UTC'] - timedelta(minutes=8)) & (
                        merged.DATETIME_UTC <= merged['START_DATETIME_UTC'] + timedelta(hours=1))
                               # &(merged.KM >= merged['KM_START'] - 5) & (merged.KM <= merged['KM_END'] + 5)
                ]

                if processed == 0:
                    merged.to_csv(
                        DATASET)
                    processed += 1
                else:
                    with open(DATASET, 'a') as f:
                        merged.to_csv(f, header=False)

                speeds_relevant = speeds_relevant[CLUSTER_SIZE:]

            merged = pd.merge(group, speeds_relevant, on=['KEY'])

            merged = merged[(merged.DATETIME_UTC >= merged['START_DATETIME_UTC'] - timedelta(minutes=8)) & (
                    merged.DATETIME_UTC <= merged['START_DATETIME_UTC'] + timedelta(hours=1))
                            #& (merged.KM >= merged['KM_START'] - 5) & (merged.KM <= merged['KM_END'] + 5)
            ]

            if processed == 0:
                merged.to_csv(
                    DATASET)
                processed += 1
            else:
                with open(DATASET, 'a') as f:
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
