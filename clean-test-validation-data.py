import pandas as pd

from datetime import timedelta

from tqdm import tqdm






if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)

    dataset = pd.read_csv("final_dataset/test.csv",
                          dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object,
                                 "STATION_ID_4": object})

    dataset.loc[(dataset.READ_INSTANT!=1), "PREC_SPEED"] = -1

    dataset.to_csv("final_dataset/test_2.csv", index=False)





    dataset = pd.read_csv("final_dataset/validation.csv",
                          dtype={"STATION_ID": object, "STATION_ID_2": object, "STATION_ID_3": object,
                                 "STATION_ID_4": object})

    dataset.loc[(dataset.READ_INSTANT!=1), "PREC_SPEED"] = -1

    dataset.to_csv("final_dataset/validation_2.csv", index=False)
