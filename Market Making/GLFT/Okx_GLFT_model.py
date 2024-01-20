import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from Constants_GLFT import *

bids_arr = np.full((TRAINING_TIME, 41), np.nan)
asks_arr = np.full((TRAINING_TIME, 41), np.nan)

def calculate_next_timestamp(timestamp):
    return (timestamp // TIME_INTERVAL + 1) * TIME_INTERVAL


def remove_nans(count):
    # Identify rows with only NaN values
    nan_rows_mask = np.isnan(count).all(axis=1)
    # Filter out rows with only NaN values
    arr_without_nan = count[~nan_rows_mask, :]
    return arr_without_nan


def cal_lambda(df, tick):
    df0 = df.reset_index()
    start_time = calculate_next_timestamp(df0.loc[0, 'Time'])
    bid_count = 0.0
    ask_count = 0.0
    bid_arr = []
    ask_arr = []
    price = df0.at[0, 'Price']
    midesh = [price]
    candle_array = np.full((86400, 4), np.nan)
    count = 0
    for index in df0.index:
        times = df0.at[index, 'Time']
        if times <= start_time:
            midesh.append(df0.at[index, 'Price'])
            if df0.at[index, 'Side'] == -1 and df0.at[index, 'Price'] <= np.round(price - tick, 1):
                bid_count += df0.at[index, 'Size']
            elif df0.at[index, 'Side'] == 1 and df0.at[index, 'Price'] >= np.round(price + tick, 1):
                ask_count += df0.at[index, 'Size']
        elif times > start_time:
            last_time = start_time
            start_time = calculate_next_timestamp(times)
            if start_time > last_time + TIME_INTERVAL:
                while last_time < start_time - TIME_INTERVAL:
                    candle_array[count] = [midesh[0], np.nanmin(midesh), np.nanmax(midesh), midesh[-1]]
                    count += 1
                    last_time += TIME_INTERVAL
                    bid_arr.append(bid_count)
                    ask_arr.append(ask_count)
                    bid_count = 0.0
                    ask_count = 0.0
            candle_array[count] = [midesh[0], np.nanmin(midesh), np.nanmax(midesh), midesh[-1]]
            count += 1
            price = df0.at[index, 'Price']
            midesh = [price]
            bid_arr.append(bid_count)
            ask_arr.append(ask_count)
            bid_count = 0.0
            ask_count = 0.0
    return bid_arr, ask_arr, remove_nans(candle_array)

def cal_mad(arr):
    arresh = arr[:, 3] - arr[:, 0]
    return np.mean(np.abs(arresh - np.mean(arresh))) * np.sqrt(3.14/2)

def config_model(symbol):
    global bids_arr, asks_arr
    date = datetime.utcnow().date()
    df = pd.read_csv(f"/home/gustakh/PycharmProjects/pythonProject/okx_{symbol}_trades_{date}.csv")
    df = df[df['Time'] >= calculate_next_timestamp(df.iloc[-1, 0]) - TRAINING_TIME * TIME_INTERVAL]
    ticks = np.arange(10, 51)
    for i in ticks:
        bid_arr, ask_arr, candle = cal_lambda(df, i / 10)
        bids_arr[:len(bid_arr), i - 10], asks_arr[:len(ask_arr), i - 10] = bid_arr, ask_arr
    return bids_arr, asks_arr, candle

def get_model(arr):
    ticks = np.arange(10, 51)
    # Fit the OLS model
    model = sm.OLS(np.log(arr), sm.add_constant(ticks / 10))
    result = model.fit()
    print("Params are: ", result.params)
    A = np.exp(result.params[0])
    k = -1 * result.params[1]
    return A, k

def compute_coeff(xi, gamma, delta, A, k):
    inv_k = np.divide(1, k)
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
    return c1, c2


if __name__ == '__main__':
    bids, asks, candle = config_model('btcusdt')
    lambda_bids = np.nanmean(bids_arr, axis=0)
    lambda_asks = np.nanmean(asks_arr, axis=0)
    a, k = get_model(lambda_asks)
    vol = cal_mad(candle[-VOL_TIME:])
    print("Volatility is: ", vol)
    print("A and k are: ", a, k)

    c1, c2 = compute_coeff(GAMMA, GAMMA, DELTA, a, k)
    half_spread = 1 * c1 + 1 / 2 * c2 * vol
    skew = c2 * vol
    print('Half_spread={}, Skew={}'.format(half_spread, skew))
