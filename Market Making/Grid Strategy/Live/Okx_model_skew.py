import numpy as np
import pandas as pd
import statsmodels.api as sm
from Constants import TIME_INTERVAL, WINDOW

import warnings
warnings.filterwarnings("ignore")


def calculate_next_timestamp(timestamp):
    return (timestamp // TIME_INTERVAL + 1) * TIME_INTERVAL

def calculate_predictions(df0):
    start_time = calculate_next_timestamp(df0.iloc[0, 0])
    imbalance = []
    mid_price = []
    sum_bid = 0.0
    sum_ask = 0.0
    for index in df0.index:
        if df0.at[index, 'Time'] < start_time:
            sum_bid += df0.at[index, 'BidVol']
            sum_ask += df0.at[index, 'AskVol']
        elif df0.at[index, 'Time'] > start_time:
            start_time += TIME_INTERVAL
            mid_price.append((df0.at[index-1, 'BestBid'] + df0.at[index-1, 'BestAsk']) / 2)
            imbalance.append(sum_bid - sum_ask)
            sum_bid = 0
            sum_ask = 0
            sum_bid += df0.at[index, 'BidVol']
            sum_ask += df0.at[index, 'AskVol']
    alpha = []
    for t in range(len(imbalance)):
        m = np.nanmean(imbalance[max(0, t + 1 - WINDOW):t + 1])
        s = np.nanstd(imbalance[max(0, t + 1 - WINDOW):t + 1])
        alpha.append(np.divide(imbalance[t] - m, s))
    # Calculating the change in midprice between this and previous interval
    midesh = np.array(mid_price)[2:] - np.roll(mid_price, 1)[2:]
    alphesh = np.roll(alpha, 1)[2:]
    return midesh, alphesh, mid_price

def linear_model(file_path):
    df0 = pd.read_csv(file_path)
    midesh, alphesh, _ = calculate_predictions(df0)

    # Create a linear regression model
    model = sm.OLS(midesh, alphesh)

    # Fit the model to the data
    return model.fit()