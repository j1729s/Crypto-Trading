from Okx_Process_Data import validate_model
import pandas as pd
import os
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
for i in range(1, len(datas)):
    acc_dict = {}
    test = next(datasets)
    for l in range(0, 5):
        for d in range(1, 6):
            acc_dict[(l, d)] = validate_model(train, test, l=l, d=d, threshold=0.1, resample=100)
    print(max(zip(acc_dict.values(), acc_dict.keys())))
    print(acc_dict)
    train = test