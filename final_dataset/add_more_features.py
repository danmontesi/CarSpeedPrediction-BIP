from datetime import timedelta

from tqdm import tqdm
import pandas as pd


holiday_list = [(1, 11), (8, 12), (24,12), (25 ,12), (26, 12), (27 ,12),
                (28 ,12), (29 ,12), (30 ,12), (31, 12), (1, 1), (2, 1),
                (3, 1), (4, 1), (5, 1), (6, 1)]

pre_holiday_list = [(31, 10), (7, 12), (23,12)]

def add_holiday(dataset):
    dataset['DATETIME_UTC'] = pd.to_datetime(dataset['DATETIME_UTC'])

    dataset['HOLIDAY'] = 0
    dataset['PRE_HOLIDAY'] = 0

    for i in tqdm(dataset.index):
        date = dataset.at[i, 'DATETIME_UTC']
        if (date.day, date.month) in holiday_list or dataset.at[i, 'WEEK_DAY'] in [6,7]:
            dataset.at[i, 'HOLIDAY'] = 1

        if (date.day, date.month) in pre_holiday_list or dataset.at[i, 'WEEK_DAY'] == 5:
            dataset.at[i, 'PRE_HOLIDAY'] = 1

    return dataset


if __name__ == '__main__':
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)

    train = pd.read_csv('merged.csv')

    train_1 = add_holiday(train)

    train_1.to_csv('merged_1.csv')

    test = pd.read_csv('../preprocess-test/test_final.csv')

    test_1 = add_holiday(test)

    test_1.to_csv('../preprocess-test/test_final_1.csv')





