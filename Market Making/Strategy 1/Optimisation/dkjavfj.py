import os
import pandas as pd
from Linear_Data_pd import linear_data
from BacktestStrategy_Imp import backtest_strategy

# Get all file names in the directories
datas = os.listdir("D:\Test")
def read_datasets():
    global datas

    # Loop through each file
    for data in datas:
        # Full path to the file
        file_path_q = os.path.join("D:\Test", data)

        # Read the dataset using pandas
        df = pd.read_csv(file_path_q)

        # Yield combined dataset
        yield df

datasets = read_datasets()
train = next(datasets)

for i in range(1,len(datas)):
    acc_dict = {}
    test = next(datasets)
    for l in range(0, 8):
        for d in range(1,9):
            tn, tt = linear_data(train, l=l, d=d), linear_data(test, l=l, d=d)
            df = backtest_strategy(tn, tt, threshold=0.01, l=l, optimise=True)
            acc_dict[(l, d)] = sum(df["Trade Cost"])
    print(max(zip(acc_dict.values(), acc_dict.keys())))
    train = test



