import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


def linear_data(data, l=0, d=1, resample=None):
    """
    Build up linear data for linear model
    :param Order_Book: Raw Order Book Data
    :param Kline_Data: Raw Kline Data
    :param freq: Frequency at which data will be resampled
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param d: the no. of DELAYS (in future) for calculating the mid-price change
    :param mpb: whether to use updated formula or original one in the paper
    :return: Dataframe with the required metrics
    """

    if resample is not None:
        # Convert the timestamp column to datetime format
        data.index = pd.to_datetime((data['Time']).astype('datetime64[ms]'), unit='ms')

        # Resample the dataset
        data = data.resample(f'{resample}ms').last().ffill()


    # Specify naming convention
    convention = {
        "BestBid": "BidDiff",
        "BidVol": "BVolDiff",
        "BestAsk": "AskDiff",
        "AskVol": "AVolDiff",
        "Volume": "VolDiff",
        "Price": "PriceDiff"
    }

    # Calculating first diferences of various columns and dropping null rows
    ldata = data[["BestBid", "BidVol", "BestAsk", "AskVol", "Volume", "Price"]].diff().rename(columns=convention)
    ldata[["BidVol", "AskVol", "Price", "BestBid", "BestAsk"]] = data[["BidVol", "AskVol", "Price", "BestBid", "BestAsk"]]
    ldata["MidPrice"] = (data["BestAsk"] + data["BestBid"]) / 2

    #Calculating Micro Price
    ldata["Micro"] = (data["BestBid"] * data["BidVol"] + data["BestAsk"] * data["AskVol"]) / (data["BidVol"] + data["AskVol"])
    ldata["Micro"] = ldata["Micro"] - data["Price"]

    # Calculating Average of MidPrice for (t,t-1) to be used while calculating MPB
    ldata["AvgMP"] = (ldata["MidPrice"] + ldata["MidPrice"].shift(1)) / 2

    ldata["Time"] = data["Time"]

    # Calculating Spread
    ldata["Spread"] = data["BestAsk"] - data["BestBid"]

    # Drop first row with Nan values
    ldata.drop(ldata.index[0], inplace=True)

    # Calculating mid-price Change for given delays
    ldata["MPC"] = ldata["MidPrice"].shift(-1).rolling(d).mean().shift(1 - d) - ldata["MidPrice"]

    # Calculating Trade Imbalance
    ldata["TI"] = np.where(ldata["VolDiff"] == 0, np.nan,
                            np.where(ldata["PriceDiff"] != 0,
                                     (ldata["Price"] + (ldata["VolDiff"] / ldata["PriceDiff"])), ldata["Price"]))
    ldata.iloc[0, ldata.columns.get_loc("TI")] = ldata.iloc[0, ldata.columns.get_loc("MidPrice")]

    ldata["TI"] = ldata["TI"].fillna(method='ffill')
    ldata["TI"] = ldata["TI"] - ldata["AvgMP"]

    # Calculating Order Imbalance
    dBid = pd.Series(np.where(ldata["BidDiff"] < 0, 0,
                              np.where(ldata["BidDiff"] == 0, ldata["BVolDiff"], ldata["BidVol"])), index=ldata.index)
    dAsk = pd.Series(np.where(ldata["AskDiff"] < 0, ldata["AskVol"],
                              np.where(ldata["AskDiff"] == 0, ldata["AVolDiff"], 0)), index=ldata.index)
    ldata["OI_(t)"] = dBid - dAsk

    if l > 0:
        # Calculating OIR and VOI for each lag by shifting data
        for i in range(1, l + 1):
            ldata[f"OI_(t-{i})"] = ldata["OI_(t)"].shift(i)

    # Dropping irrelevant columns
    ldata = ldata.drop(columns=ldata.columns[:9])

    # Return dataframe with required metrics
    return ldata.dropna()


def linear_model(train_data, l=1):
    """
    Build up linear model
    :param train_data: Training Dataset
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :return: Linear model or coefficient
    """

    # Build the explanatory variables
    df = pd.DataFrame({'y': train_data["MPC"],
                       'OI': train_data["OI_(t)"] / train_data['Spread'],
                       **{f'OI{i}': train_data[f"OI_(t-{i})"] / train_data['Spread'] for i in range(1, l + 1)}})

    # Build the linear model using OLS
    model = sm.OLS(df['y'], sm.add_constant(df.drop('y', axis=1))).fit()

    # Return model
    return model

def calculate_metrics(true_labels, predicted_labels):
    # Calculate TP, TN, FP, FN
    TP = np.sum((true_labels == 1) & (predicted_labels == 1))
    TN = np.sum((true_labels == -1) & (predicted_labels == -1))
    FP = np.sum((true_labels == -1) & (predicted_labels == 1))
    FN = np.sum((true_labels == 1) & (predicted_labels == -1))

    # The ratio of FP0 to FN0 should be optimised to be as close to zero. Because most of these are
    # position are opposing orders on the same price, so it gives us volume with loss
    TP0 = np.sum((true_labels == 0) & (predicted_labels == 0))
    FP0 = np.sum((true_labels == 0) & ((predicted_labels == 1) | (predicted_labels == -1)))
    FN0 = np.sum(((true_labels == 1) | (true_labels == -1)) & (predicted_labels == 0))

    # Calculate precision, recall, and F1 score
    precision0 = TP0 / (TP0 + FP0) if (TP0 + FP0) != 0 else 0
    recall0 = TP0 / (TP0 + FN0) if (TP0 + FN0) != 0 else 0
    f1_score0 = 2 * (precision0 * recall0) / (precision0 + recall0) if (precision0 + recall0) != 0 else 0

    # Calculate precision, recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Create a dictionary to store the results
    metrics_dict = {
        "True Positives": TP,
        "True Negatives": TN,
        "False Positives": FP,
        "False Negatives": FN,
        "True Positives for Zeros (TP0)": TP0,
        "False Positives for Zeros (FP0)": FP0,
        "False Negatives for Zeros (FN0)": FN0,
        "FP0/FN0 ratio for Zeros": FP0/FN0,
        "F1 Score": f1_score,
        "F1 Score for Zeros": f1_score0,
        "True Signal": np.sum((true_labels == 1) | (true_labels == -1)),
        "Predicted Signal": np.sum((predicted_labels == 1) | (predicted_labels == -1))

    }

    return metrics_dict

def validate_model(tn_data, tt_data, l=1, d=2, threshold=0.1, resample=None):
    """
    Prints out metrics for model evaluation
    :param train_data: Training Dataset
    :param test_data: Test data for prediction
    :param l: the no. of LAGS for OI
    :param threshold: trading threshold
    """
    train_data = linear_data(tn_data, l=l, d=d, resample=resample)
    test_data = linear_data(tt_data, l=l, d=d, resample=resample)

    # Retrieve trained model
    model = linear_model(train_data, l=l)

    df = pd.DataFrame({'y': test_data["MPC"],
                       'OI': test_data["OI_(t)"] / test_data['Spread'],
                       **{f'OI{i}': test_data[f"OI_(t-{i})"] / test_data['Spread'] for i in range(1, l + 1)}})

    y_pred = model.predict(sm.add_constant(df.drop('y', axis=1)))

    # Converting to multinomial classifier
    y_true = np.where(df['y'] > threshold, 1, np.where(df['y'] < -threshold, -1, 0))
    y_pred = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))

    # returns accuracy score for the predictions
    return accuracy_score(y_true, y_pred), calculate_metrics(y_true, y_pred)