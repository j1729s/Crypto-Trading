import argparse
import numpy as np
import pandas as pd


def join_data(Order_Book, Kline_Data, freq='100ms'):
    """
    Joins Raw data into usable dataframes
    :param Order_Book: Raw Order Book Data
    :param Kline_Data: Raw Kline Data
    :param freq: Frequency at which data will be resampled
    :return: Joined dataframe with no gaps
    """
    
    # Define convention for coloumn names
    convention = {
                  'ask_price': 'BestAsk',
                  'bid_price': 'BestBid',
                  'ask_amount': 'AskVol',
                  'bid_amount': 'BidVol',
                  'amount':'Volume',
                  'price':'Price'
                   }
    
    # Drop unnecessary columns and sort by timestamp
    Order_Book=Order_Book.sort_values(by='timestamp').drop(["local_timestamp", "exchange", "symbol"], axis=1)
    Kline_Data=Kline_Data.sort_values(by='timestamp')
    
    # Group Trade data by timestamp to aggregate trades and join
    ## Price is the mean price for a timestamp while volume is the sum
    Kline_Data = (Kline_Data[['timestamp', 'amount']].groupby('timestamp').sum()).join(Kline_Data[['timestamp', 'price']].groupby('timestamp').last()).reset_index()
    
    # Calculate volume since day start as cumulative sum
    Kline_Data['amount'] = Kline_Data['amount'].cumsum()
    
    # Converting timestamps from unix to datetime
    Order_Book.index=pd.to_datetime((Order_Book['timestamp']/1e3).astype('datetime64[ms]').rename('Time'), unit='ms')
    Kline_Data.index=pd.to_datetime((Kline_Data['timestamp']/1e3).astype('datetime64[ms]').rename('Time'), unit='ms')
    
    # Drop unix timestamp columns
    Order_Book.drop("timestamp", axis=1, inplace=True)
    Kline_Data.drop("timestamp", axis=1, inplace=True)
    
    # Upsampling to given 'freq' data incase of gaps
    Order_Book = Order_Book.resample(freq).ffill().dropna()
    Kline_Data = Kline_Data.resample(freq).ffill().dropna()
    
    # Returns joined dataset
    return Order_Book.join(Kline_Data).dropna().rename(columns=convention)
    

def linear_data(Order_Book, Kline_data, freq='100ms', l=0, d=2, mpb='original'):
    """
    Build up linear data for linear model
    :param Order_Book: Raw Order Book Data 
    :param Kline_Data: Raw Kline Data
    :param freq: Frequency at which data will be resampled
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param d: the no. of DELAYS (in future) for calculating the mid price change
    :param mpb: whether to use updated formula or original one in the paper
    :return: Dataframe with the required metrics
    """
    
    # Joining raw data into a single dataframe and adding turnover 
    data = join_data(Order_Book, Kline_data, freq)
    data['Turnover'] = data['Price']*data['Volume']
    
    # Specify naming convention
    convention = {
              "BestBid":"BidDiff", 
              "BidVol":"BVolDiff",
              "BestAsk": "AskDiff",
              "AskVol": "AVolDiff",
              "Turnover": "TurnDiff",
              "Volume": "VolDiff",
              "Price": "PriceDiff"
                }
    
    # Calculating first diferrences of various columns and dropping null rows
    ldata = data[["BestBid", "BidVol", "BestAsk", "AskVol", "Volume", "Turnover", "Price"]].diff().rename(columns=convention)
    ldata[["BidVol", "AskVol", "Price", "BestBid", "BestAsk"]] = data[["BidVol", "AskVol", "Price", "BestBid", "BestAsk"]]
    ldata["MidPrice"] = (data["BestAsk"] + data["BestBid"])/2
    
    # Calculating Average of MidPrice for (t,t-1) to be used while calculating MPB
    ldata["AvgMP"] = (ldata["MidPrice"] + ldata["MidPrice"].shift(1))/2
    
    # Calculating Spread
    ldata["Spread"] = data["BestAsk"] - data["BestBid"]
    
    # Drop first row with Nan values
    ldata.drop(ldata.index[0], inplace=True)
    
    # Calculating Mid Price Change for given delays
    ldata["MPC"] = ldata["MidPrice"].shift(-1).rolling(d).mean().shift(1-d) - ldata["MidPrice"]
    
    # Calculating Mid Price Basis
    if mpb == 'updated':
        ldata["MPB"] = np.where(ldata["VolDiff"] == 0, np.nan, 
                                np.where(ldata["PriceDiff"] != 0, (ldata["Price"] +(ldata["VolDiff"]/ldata["PriceDiff"])), ldata["Price"]))
        ldata.iloc[0, ldata.columns.get_loc("MPB")] = ldata.iloc[0, ldata.columns.get_loc("MidPrice")]
        ldata["MPB"] = ldata["MPB"].fillna(method='ffill')
        ldata["MPB"] = ldata["MPB"] - ldata["AvgMP"]
    else:
        ldata["MPB"] = np.where(ldata["VolDiff"] != 0, (ldata['TurnDiff']/ldata['VolDiff']), np.nan)
        ldata.iloc[0, ldata.columns.get_loc("MPB")] = ldata.iloc[0, ldata.columns.get_loc("MidPrice")]
        ldata["MPB"] = ldata["MPB"].fillna(method='ffill')
        ldata["MPB"] = ldata["MPB"] - ldata["AvgMP"]
    
    # Calculating OIR
    ldata["OIR_(t)"] = (ldata["BidVol"] - ldata["AskVol"])/(ldata["BidVol"] + ldata["AskVol"])
    
    # Calculating VOI
    dBid = pd.Series(np.where(ldata["BidDiff"] < 0, 0, 
                              np.where(ldata["BidDiff"] == 0, ldata["BVolDiff"], ldata["BidVol"])), index=ldata.index)
    dAsk = pd.Series(np.where(ldata["AskDiff"] < 0, ldata["AskVol"], 
                              np.where(ldata["AskDiff"] == 0, ldata["AVolDiff"], 0)), index=ldata.index)
    ldata["VOI_(t)"] = dBid - dAsk
    
    # Calculating OIR and VOI for each lag by shifting data
    for i in range(1, l+1):
        ldata[f"OIR_(t-{i})"] = ldata["OIR_(t)"].shift(i)
        ldata[f"VOI_(t-{i})"] = ldata["VOI_(t)"].shift(i)
        
    # Dropping irrelevant columns
    ldata = ldata.drop(columns=ldata.columns[:9])
    ldata = ldata.drop(columns=["AvgMP"])
    
    # Return dataframe with required metrics
    return ldata.dropna()


if __name__ == '__main__':
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Build up linear data for linear model')

    # Add arguments
    parser.add_argument('order_book_path', type=str, help='Path to raw Order Book data CSV file')
    parser.add_argument('kline_data_path', type=str, help='Path to raw Kline Data CSV file')
    parser.add_argument('--freq', default='100ms', help='Resampling frequency (default: 100ms)')
    parser.add_argument('-l', type=int, default=5, help='The number of lags for VOI and OIR determined by the ACF Plot (default=0)')
    parser.add_argument('-d', type=int, default=20, help='The number of delays (in future) for calculating the mid price change (default=2)')
    parser.add_argument('--mpb', choices=['original', 'updated'], default='original', help='Method for calculating the midpoint price (default: original)')

    # Parse the arguments
    args = parser.parse_args()
    
    # Read the data into a dataframe
    OrderBook = pd.read_csv(args.order_book_path)
    KlineData = pd.read_csv(args.kline_data_path)

    # Call the function with the parsed arguments
    data = linear_data(OrderBook, KlineData, args.freq, args.l, args.d, args.mpb)
    print(data)
