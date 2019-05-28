import pandas as pd

from datetime import timedelta

from tqdm import tqdm

CLUSTER_SIZE = 20000
DATASET = "bip_assignment/dataset_with_all_km.csv"


def preprocess():
    def categorizeData(data):
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

    def convert_weather(dataset):

        dataset['WEATHER'] = (dataset['WEATHER']).apply(
            lambda x: categorizeData(x))

        return dataset

    dataset = pd.read_csv("bip_assignment/dataset_3.csv")

    dataset = convert_weather(dataset)

    dataset.to_csv("bip_assignment/dataset_4.csv", index=False)


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)

    preprocess()
