import pandas as pd

from datetime import timedelta

from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


def preprocess():
    speeds_df = pd.read_csv("output.csv",
                            dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object,
                                   "STATION_ID_4": object})
    speeds_df['SPEED_AVG'] = speeds_df['PREDICTION']
    speeds_df=speeds_df.drop(['PREDICTION'], axis=1)

    #print(speeds_df.head(25))


    #touple_df_1 = set(zip(speeds_df.KEY, speeds_df.KM))

    speeds_df['DATETIME_UTC'] = pd.to_datetime(speeds_df['DATETIME_UTC'])
    #print(speeds_df[(speeds_df.KEY == 0) & (speeds_df.KM == 407) & (speeds_df.DATETIME_UTC >= "2019-01-12") & (speeds_df.DATETIME_UTC <= "2019-01-13")].head(200))

    evaluation = pd.read_csv("speeds_evaluation__only_datetimes__2019.csv")
    #touple_eval=set(zip(evaluation.KEY, evaluation.KM))

    #print(touple_eval-touple_df_1)

    print(evaluation.shape[0])
    evaluation['DATETIME_UTC'] = pd.to_datetime(evaluation['DATETIME_UTC'])


    speeds_df['PREDICTION_STEP'] = speeds_df['READ_INSTANT'] - 1
    speeds_df=speeds_df.drop(['READ_INSTANT'], axis=1)
    speeds_df = speeds_df[['KEY', 'KM', 'DATETIME_UTC', 'PREDICTION_STEP', 'SPEED_AVG']]
    speeds_df=speeds_df.groupby(['KEY', 'KM', 'DATETIME_UTC', 'PREDICTION_STEP'], as_index=False).mean()


    #print(speeds_df.shape[0])
    #speeds_df = speeds_df.drop_duplicates()
    #print(speeds_df.shape[0])
    #speeds_df["SPEED_AVG"] = -1




    evaluation = evaluation.merge(speeds_df, on=['KEY','KM','DATETIME_UTC','PREDICTION_STEP'], how='left')


    evaluation=evaluation.sort_values(['KEY', 'KM', 'DATETIME_UTC'])
    print(evaluation.shape[0])

    print(evaluation.head(25))

    evaluation.to_csv("submission.csv", index=False)



if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)
    preprocess()
