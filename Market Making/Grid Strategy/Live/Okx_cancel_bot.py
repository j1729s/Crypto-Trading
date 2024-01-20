import csv
import schedule
import numpy as np
from time import sleep
import okx.Trade as Trade
import okx.Account as Account
from datetime import datetime
from multiprocessing import Pool
from Okx_Secret import API_KEY, API_SECRET, API_PASS
import sys
from contextlib import contextmanager

CHECK_LAG = 3
NUM_PROCESSES = 1

accountAPI = Account.AccountAPI(API_KEY, API_SECRET, API_PASS, False, '0')
tradeAPI = Trade.TradeAPI(API_KEY, API_SECRET, API_PASS, False, '0')

new_date = datetime.utcnow().date()
#file = open(f'Metrics_{new_date}.csv', 'w', newline='')
#writer = csv.writer(file)
#writer.writerow(["Time", "NetPosition", "OpenOrders", "BalanceUSD", "LiquidationPrice", "MarkPrice", "Mmr", "AvailableEq"])

def to_float(value):
    try:
        # Attempt to convert the string to a float
        result = float(value)
    except ValueError:
        # Handle the case where the conversion fails
        result = np.nan
    return result

def check_position():
    try:
        x = accountAPI.get_positions(instId='BTC-USDT-SWAP')
        data = x['data']
        for dat in data:
            if dat['mgnMode'] == 'cross':
                times = to_float(dat['uTime'])
                pos = to_float(dat['pos'])
                liqpx = to_float(dat['liqPx'])
                markpx = to_float(dat['markPx'])
                mmr = to_float(dat['mmr'])
                return [times, pos, liqpx, markpx, mmr]
        else:
            return [np.nan, np.nan, np.nan, np.nan, np.nan]
    except Exception as E:
        print(f"Exception while fetching position: {E}")


def check_orders():
    try:
        len_ord = 0
        ordId = ''
        while True:
            x = tradeAPI.get_order_list(instId="BTC-USDT-SWAP", after=ordId)
            len_data = len(x['data'])
            len_ord += len_data
            if len_data == 100:
                ordId = x['data'][-1]['ordId']
            elif len_data < 100:
                break
        return float(len_ord)
    except Exception as E:
        print(f"Exception while checking orders: {E}")

def check_balance():
    try:
        x = accountAPI.get_account_balance()
        balance = x['data'][0]['details'][0]['eqUsd']
        available = x['data'][0]['details'][0]['availEq']
        return [round(float(balance), 2), float(available)]
    except Exception as E:
        print(f"Exception while fetching balance: {E}")

# Wrapper function for each task
def task_wrapper(func):
    return func()

def paralell_check():
    global writer
    # Create a Pool of processes
    with Pool(NUM_PROCESSES) as pool:
        # Map the wrapper function to the pool for parallel execution
        results = pool.map_async(task_wrapper, [check_position, check_orders, check_balance])
        # Get the results
        results = results.get()
    print(['Time', 'Position', 'Open Orders', 'Balance', 'Liquidation Price', 'Mark Price', 'Mmr', 'AvailableEq'])
    try:
        print([results[0][0], results[0][1], results[1], results[2][0], results[0][2], results[0][3], results[0][4], results[2][1]])
        #writer.writerow([results[0][0], results[0][1], results[1], results[2], results[0][2], results[0][3]])
    except Exception as E:
        print(f"An exception occurred while collecting the results: {E}")


def main():
    schedule.every(CHECK_LAG).seconds.do(paralell_check)
    while True:
        schedule.run_pending()
        sleep(CHECK_LAG)

if __name__=="__main__":
    main()