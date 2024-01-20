import os
import pandas as pd
from Okx_Process_Data import linear_data
from Okx_Backtest_Taker import backtest_taker
import warnings
warnings.filterwarnings("ignore")

# Place path to folder below
PATH = "/media/gustakh/DATA/Test/"
# Get all file names in the directory
datas = sorted(os.listdir(PATH))
def read_datasets():
    global datas
    # Loop through each file
    for data in datas:
        file_path_q = os.path.join(PATH, data)
        print(data)
        # Read the dataset using pandas
        df1 = pd.read_csv(file_path_q)

        # Yield combined dataset
        yield df1

datasets = read_datasets()
train = next(datasets)

for i in range(1,len(datas)):
    acc_dict = {}
    test = next(datasets)
    for l in range(0, 4):
        for d in range(1, 5):
            tn, tt = linear_data(train, l=l, d=d, resample=100), linear_data(test, l=l, d=d, resample=100)
            df = backtest_taker(tn, tt, threshold=0.1, l=l, optimise=True)
            acc_dict[(l, d)] = sum(df["Trade Cost"]), len(df)
    print(max(zip(acc_dict.values(), acc_dict.keys())))
    print(acc_dict)
    train = test