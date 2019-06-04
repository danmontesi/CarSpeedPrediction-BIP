import pandas as pd

from datetime import timedelta

from tqdm import tqdm


def preprocess():

    speeds_df = pd.read_csv("bip_assignment/dataset_7.csv", dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object, "STATION_ID_4": object})
    sup_df = pd.read_csv("superfluous_data.csv")

    speeds_df = speeds_df.merge(sup_df, on=['KEY', 'KM', 'TIME_INTERVAL'])
    speeds_df.to_csv("bip_assignment/dataset_8.csv", index=False)


if __name__ == "__main__":
    preprocess()
