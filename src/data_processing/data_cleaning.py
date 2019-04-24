import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.misc import get_data_file_path
from sklearn.preprocessing import LabelEncoder

class FlightDataImport(object):

    def __init__(self, filename, columns=None):
        self.__file = get_data_file_path(filename)
        self.__main_struct = None

        self.data_clean = None
        self.data_normal = None

        self.__file_import()
        if columns != None:
            self.__data_drop(columns)
        self.__normal_data()

    def __file_import(self):
        self.__main_struct = pd.read_csv(self.__file)

    def __data_drop(self, columns):
        self.data_clean = self.__main_struct.drop(columns, axis=1)

    def __normal_data(self):
        mean = self.data_clean.mean(axis=0)
        meanE = self.data_clean['Error'].mean(axis=0)
        stdE = self.data_clean['Error'].std(axis=0)
        flight_data = self.data_clean - mean
        std = self.data_clean.std(axis=0)

        self.data_normal = flight_data / std

    def get_data(self):
        return self.__main_struct

    def get_clean(self):
        return  self.data_clean

    def get_normal(self):
        return self.data_normal

if __name__ == "__main__":

    datafile = 'Arriving Data.csv'
    FlightFile = FlightDataImport(datafile, columns=['AircraftType','BlockArrTime','DepCountry',
                                       'BlockArrDate','NewMsgTime','MsgDate', 'NewMsgDate', 'BlockDepDate'])

    arrivals_data = FlightFile.get_data()
    arrivals_clean = FlightFile.get_clean()
    arrivals_norm = FlightFile.get_normal()