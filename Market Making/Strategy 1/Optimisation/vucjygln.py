from Linear_Data_pd import validate_model
import pandas as pd
import os

# Get all file names in the directory
datas = os.listdir("D:/test")
def read_datasets():
    global datas
    # Loop through each file
    for data in datas:
        file_path_q = os.path.join("D:/test", data)

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
            acc_dict[(l, d)] = validate_model(train, test, l=l, d=d, threshold=0.01)
    print(max(zip(acc_dict.values(), acc_dict.keys())))
    train = test
