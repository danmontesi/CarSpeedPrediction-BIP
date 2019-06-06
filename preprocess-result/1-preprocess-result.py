import pandas as pd

from datetime import timedelta

from tqdm import tqdm

CLUSTER_SIZE = 20000
DATASET = "2019_final.csv"


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



def simplifyWeather(data):
    if data == "Quasi Sereno":
        return "QuasiSereno"
    if data == "Debole Pioggia":
        return "DebolePioggia"
    if data == "Foschia":
        return "Foschia"
    if data == "Fumo":
        return "Fumo"
    if data == "Temporale con Debole Pioggia":
        return "DebolePioggia"
    if data == "Debole Pioggerella":
        return "DebolePioggia"
    if data == "Tempesta di Polvere":
        return "Fumo"
    if data == "Rovescio Nelle Vicinanze con Pioggia":
        return "DebolePioggia"
    if data == "Nelle Vicinanze Tempesta di Polvere":
        return "Fumo"
    if data == "Rovescio con Debole Pioggia":
        return "DebolePioggia"
    if data == "Sottili Banchi di Nebbia":
        return "Foschia"
    if data == "Debole Neve":
        return "Neve"
    if data == "Pioggia Gelata":
        return "FortePioggia"
    if data == "Quasi sereno":
        return "QuasiSereno"
    if data == "Forte Pioggia":
        return "FortePioggia"
    if data == "Banchi di Nebbia":
        return "Foschia"
    if data == "Parzialmente Nebbia":
        return "Foschia"
    if data == "Rovescio con Forte Pioggia":
        return "FortePioggia"
    if data == "Gelata con Foschia":
        return "Foschia"
    if data == "Nelle Vicinanze Nebbia":
        return "Foschia"
    if data == "Rovescio con Grandine Piccola":
        return "DebolePioggia"
    if data == "Debole Pioggia e Neve":
        return "DebolePioggia"
    if data == "weather.TSSN":
        return ""
    return 'other';




def preprocess():
    def compute_weather(grouped, weather_df):

        for name, group in (grouped):
            print(name)

            weather_relevant = weather_df[weather_df.ID == name]

            for i in tqdm(group.index):
                # line=group[group.index == i]

                weather_very_relevant = weather_relevant#[
                #    (group.at[i, 'DATETIME_UTC'] >= weather_relevant.DATETIME_UTC - timedelta(hours=12))
                #    & (group.at[i, 'DATETIME_UTC'] <= weather_relevant.DATETIME_UTC + timedelta(hours=12))]
                weather_very_relevant["TIME_WEATHER_DELTA"] = abs(
                    weather_relevant.DATETIME_UTC - group.at[i, 'DATETIME_UTC'])
                # print(weather_very_relevant.head())
                if weather_very_relevant[["TIME_WEATHER_DELTA"]].shape[0] >0:
                    index = int(weather_very_relevant[["TIME_WEATHER_DELTA"]].idxmin())
                    # print(index)
                    if (weather_very_relevant.at[index, "TIME_WEATHER_DELTA"]<=timedelta(hours=12)):
                        merged.loc[i, 'TEMPERATURE'] = weather_very_relevant.at[index, "TEMPERATURE"]
                        merged.loc[i, 'MAX_TEMPERATURE'] = weather_very_relevant.at[index, "MAX_TEMPERATURE"]
                        merged.loc[i, 'MIN_TEMPERATURE'] = weather_very_relevant.at[index, "MIN_TEMPERATURE"]
                        merged.loc[i, 'WEATHER'] = weather_very_relevant.at[index, "WEATHER"]
                        merged.loc[i, 'DATETIME_UTC_WEATHER'] = weather_very_relevant.at[index, "DATETIME_UTC"]

    events_df = pd.read_csv("events_2019.csv")

    events_df['START_DATETIME_UTC'] = pd.to_datetime(events_df['START_DATETIME_UTC'])
    events_df['END_DATETIME_UTC'] = pd.to_datetime(events_df['END_DATETIME_UTC'])

    events_df['APPROX_TIME'] = (events_df['START_DATETIME_UTC'] + timedelta(seconds=1)).dt.ceil('15min')





    speeds_df = pd.read_csv("speeds_2019.csv", usecols=['KEY', 'KM'])

    speeds_df=speeds_df.drop_duplicates() #only touple unique KEY KM

    merged = events_df.merge(speeds_df, on=["KEY"])
    merged['tmp'] = 1



    instant = pd.DataFrame(data = [1, 2, 3, 4], columns=['READ_INSTANT'])
    instant['tmp'] = 1

    merged=merged.merge(instant, on=["tmp"])

    merged=merged.drop(["tmp"], axis=1)



    merged["DATETIME_UTC"] = merged["APPROX_TIME"] + (timedelta(minutes=15) * (merged["READ_INSTANT"] -1))

    merged['DISTANCE_FROM_POINT'] = merged['KM'] - merged['KM_START']
    merged['LENGTH_KM'] = merged['KM_END'] - merged['KM_START']

    merged['WEEK_DAY'] = merged['DATETIME_UTC'].apply(lambda x: x.weekday())
    merged['TIME_INTERVAL'] = merged['DATETIME_UTC'].apply(categorizeData)

    merged['DELTA_TIME_FROM_START'] = (merged['DATETIME_UTC'] - merged['START_DATETIME_UTC']).apply(lambda x: round(x.seconds / 60, 1))

    road_df = pd.read_csv("../bip_assignment/sensors.csv")

    merged = merged.merge(road_df, on=["KEY", "KM"], how='left')

    print(merged.head(50))

    merged.to_csv("PARTIAL.CSV")


#   STATIONS
    #print("STATIONS")
    #weather_list = pd.DataFrame(columns=["KEY", "KM", "STATION_ID", "STATION_ID_2"])
    #with open("../bip_assignment/distances.csv") as f:
    #    for line in tqdm(f):
    #        splitted = line.split("|")
    #        chiavina = splitted[0].split(",")
    #        valorino = splitted[1].split(",")
    #        KEY = chiavina[0]
    #        KM = chiavina[1]
    #        STATION_ID = valorino[0]
    #        STATION_ID_2 = ""
    #        if len(valorino) > 2:
    #            STATION_ID_2 = valorino[2]
    #        STATION_ID_3 = ""
    #        if len(valorino) > 4:
    #            STATION_ID_3 = valorino[4]
    #        STATION_ID_4 = ""
    #        if len(valorino) > 6:
    #            STATION_ID_4 = valorino[6]
#
    #        if KM.isdigit() and KEY.isdigit():
    #            weather_list = weather_list.append(
    #                {'KM': int(KM), 'KEY': int(KEY), 'STATION_ID': STATION_ID, 'STATION_ID_2': STATION_ID_2,
    #                 'STATION_ID_3': STATION_ID_3, 'STATION_ID_4': STATION_ID_4}, ignore_index=True)
#
    ##print(weather_list.head())
#
    #weather_list.to_csv("distances_weather_pandas.csv", index=False)

    weather_list=pd.read_csv("distances_weather_pandas.csv")

    merged = merged.merge(weather_list, on=["KEY", "KM"], how='left')

    #WEATHER
    print("WEATHER")
    weather_df = pd.read_csv("weather_2019.csv")

    merged['TEMPERATURE'] = 0
    merged['MAX_TEMPERATURE'] = 0
    merged['MIN_TEMPERATURE'] = 0
    merged['WEATHER'] = 0
    merged['DATETIME_UTC_WEATHER'] = 0

    weather_df['DATETIME_UTC'] = pd.to_datetime(weather_df['DATETIME_UTC'])
    # grouped_dataset = dataset.groupby('STATION_ID')

    # print(dataset[dataset['STATION_ID'] == np.nan].head())

    # compute_weather(grouped_dataset, weather_df)

    print("Fixing missing values first try...")
    missing_values = merged[merged.WEATHER == 0]
    print(missing_values.shape[0])
    grouped_dataset = missing_values.groupby('STATION_ID')
    compute_weather(grouped_dataset, weather_df)

    print("Fixing missing values first try...")
    missing_values = merged[merged.WEATHER == 0]
    print(missing_values.shape[0])
    grouped_dataset = missing_values.groupby('STATION_ID_2')
    compute_weather(grouped_dataset, weather_df)

    print("Fixing missing values second try...")
    missing_values = merged[merged.WEATHER == 0]
    print(missing_values.shape[0])
    grouped_dataset = missing_values.groupby('STATION_ID_3')
    compute_weather(grouped_dataset, weather_df)

    print("Fixing missing values third try...")
    missing_values = merged[merged.WEATHER == 0]
    print(missing_values.shape[0])
    grouped_dataset = missing_values.groupby('STATION_ID_4')
    compute_weather(grouped_dataset, weather_df)

    print("Final missing:")
    missing_values = merged[merged.WEATHER == 0]
    print(missing_values.shape[0])

    merged['WEATHER'] = (merged['WEATHER']).apply(
        lambda x: simplifyWeather(x))



    #ATTACH PREC SPEED
    print("ATTACHING SPEED")
    dataset_orig = pd.read_csv("speeds_2019.csv")
    dataset_orig['DATETIME_UTC'] = pd.to_datetime(dataset_orig['DATETIME_UTC'])
    merged['PREC_SPEED'] = -1

    grouped_dataset = merged[merged['READ_INSTANT'] == 1].groupby('KEY')

    for name, group in tqdm(grouped_dataset):
        reduced_orig = dataset_orig[dataset_orig['KEY'] == name]

        for i in tqdm(group.index):
            interesting = reduced_orig[(reduced_orig['KM'] == group.at[i, 'KM']) & (
                        reduced_orig['DATETIME_UTC'] == group.at[i, 'DATETIME_UTC'] - timedelta(minutes=15))]
            # print(interesting.head())
            if interesting.shape[0] > 0:
                # print(interesting.SPEED_AVG)
                merged.loc[i, 'PREC_SPEED'] = float(interesting.SPEED_AVG.ravel()[0])
                # print(interesting.SPEED_AVG.ravel()[0])











    merged.to_csv(DATASET,  index=False)



if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)
    preprocess()
