import pandas as pd

from datetime import timedelta

from tqdm import tqdm


def preprocess():
    speeds_df = pd.read_csv("2019_final.csv",
                            dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object,
                                   "STATION_ID_4": object})
    sup_df = pd.read_csv("../superfluous_data.csv")

    speeds_df['DATETIME_UTC'] = pd.to_datetime(speeds_df['DATETIME_UTC'])
    speeds_df['START_DATETIME_UTC'] = pd.to_datetime(speeds_df['START_DATETIME_UTC'])
    speeds_df['END_DATETIME_UTC'] = pd.to_datetime(speeds_df['END_DATETIME_UTC'])
    speeds_df['APPROX_TIME'] = pd.to_datetime(speeds_df['APPROX_TIME'])

    speeds_df = speeds_df.merge(sup_df, on=['KEY', 'KM', 'WEEK_DAY', 'TIME_INTERVAL'],  how='left')

    speeds_df['DELTA_TIME'] = speeds_df['DATETIME_UTC'] - speeds_df['APPROX_TIME']

    speeds_df['DISTANCE_FROM_EVENT_MINUTES'] = (speeds_df['DATETIME_UTC'] - speeds_df['START_DATETIME_UTC']).apply(
        lambda x: x.total_seconds() / 60)

    speeds_df = speeds_df[
        ['EVENT_DETAIL', 'EVENT_TYPE', 'KEY', 'DATETIME_UTC', 'KM', 'DISTANCE_FROM_POINT', 'LENGTH_KM', 'WEEK_DAY',
         'TIME_INTERVAL', 'DELTA_TIME_FROM_START', 'EMERGENCY_LANE', 'LANES', 'ROAD_TYPE', 'TEMPERATURE',
         'MAX_TEMPERATURE',
         'MIN_TEMPERATURE', 'WEATHER', 'APPROX_TIME', 'DELTA_TIME', 'DISTANCE_FROM_EVENT_MINUTES', 'READ_INSTANT',
         'PREC_SPEED', 'AVG_ALL_DATASET', 'STD_DEV_ALL_DATASET']]

    speeds_df.to_csv("2019_final_2.csv", index=False)


if __name__ == "__main__":
    preprocess()
