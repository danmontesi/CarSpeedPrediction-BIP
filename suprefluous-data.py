import pandas as pd

from datetime import timedelta

from tqdm import tqdm


def categorizeData(data):
    time = data.hour
    if (time >= 6 and time <= 10):
        return "EarlyMorning"
    if (time >= 11 and time <= 14):
        return "LateMorning"
    if (time >= 15 and time <= 17):
        return "Afternoon"
    if (time >= 18 and time <= 21):
        return "Evening"
    if (time >= 22 and time <= 24):
        return "LateEvening"
    if (time >= 0 and time <= 5):
        return "Night"


def preprocess():

    speeds_df = pd.read_csv("bip_assignment/speeds_train.csv")

    speeds_df['DATETIME_UTC'] = pd.to_datetime(speeds_df['DATETIME_UTC'])
    speeds_df['TIME_INTERVAL'] = speeds_df['DATETIME_UTC'].apply(categorizeData)

    grouped_events = speeds_df.groupby(['KEY', 'KM', 'TIME_INTERVAL'])

    dataset = pd.DataFrame(columns=['KEY', 'KM', 'TIME_INTERVAL', 'AVG_ALL_DATASET', 'STD_DEV_ALL_DATASET'])



    for name, group in tqdm(grouped_events):
        #print(name)
        dataset=dataset.append({'KEY':name[0], 'KM':name[1], 'TIME_INTERVAL':name[2], 'AVG_ALL_DATASET':group.SPEED_AVG.mean(), 'STD_DEV_ALL_DATASET':group.SPEED_AVG.std()}, ignore_index=True)


    dataset.to_csv("superfluous_data.csv", index=False)



if __name__ == "__main__":
    preprocess()
