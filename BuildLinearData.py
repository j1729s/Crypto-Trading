import argparse
import numpy as np
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler


def join_data(Order_Book, Kline_Data, freq='250ms'):
    """
    Joins Raw data into usable dataframes
    :param Order_Book: Raw Order Book Data (250ms ticks)
    :param Kline_Data: Raw Kline Data (250ms ticks)
    :return: Joined dataframe with no gaps
    """
    
    if "Unnamed: 0" in Order_Book.columns and "Unnamed: 0" in Kline_Data.columns:
        # Removing unwanted column
        Order_Book = Order_Book.drop("Unnamed: 0", axis=1)
        Kline_Data = Kline_Data.drop("Unnamed: 0", axis=1)
    
    # Smoothening sharp jumps (>10 percent) in the Bid and Ask price
    avgP = Kline_Data["Price"].mean()
    dummy = Order_Book[["BestBid", "BestAsk"]].diff()
    index_b = Order_Book[["BestBid"]][abs(dummy["BestBid"]) > 0.1*avgP].index
    index_a = Order_Book[["BestAsk"]][abs(dummy["BestAsk"]) > 0.1*avgP].index
    
    ## Vectorization possible but harder to handle consecutive jumps therefore not efficient
    for i in index_b:
        if i+1 in index_b:
            Order_Book.loc[i, "BestBid"] = Order_Book.loc[i-1, "BestBid"]

    for i in index_a:
        if i+1 in index_a:
            Order_Book.loc[i, "BestAsk"] = Order_Book.loc[i-1, "BestAsk"]
    
    # Removing NaN values from Order Book Data
    Order_Book[["BestBid", "BestAsk"]] = Order_Book[["BestBid", "BestAsk"]].fillna(method='ffill')
    
    # Adding Turnover to Kline Data
    Kline_Data["Turnover"] = Kline_Data["Volume"]*Kline_Data["Price"] 
    #Turnover = Total Value Traded = Value of a Contract * No. of Contracts traded = Price * Volume
    
    # Matching the timestamps of both datasets
    if Order_Book["Timestamp"][0] > Kline_Data["Timestamp"][0]:
        
        # We match the index with the minimum difference from the first timestamp of the later data
        index = Kline_Data.index[abs(Kline_Data["Timestamp"] - Order_Book["Timestamp"][0]) 
                                 == min(abs(Kline_Data["Timestamp"] - Order_Book["Timestamp"][0]))].to_list()[0]
        
        # Calculate the difference in time and drop the unnecessary rows
        diff = Kline_Data["Timestamp"][index] - Order_Book["Timestamp"][0]
        Kline_Data = Kline_Data.drop(Kline_Data.index[0:index]).reset_index(drop=True)
        
        # Match the times by substracting the difference
        Kline_Data["Timestamp"] = Kline_Data["Timestamp"] - diff
        
    else:
        # We match the index with the minimum difference from the first timestamp of the later data
        index = Order_Book.index[abs(Order_Book["Timestamp"] - Kline_Data["Timestamp"][0]) 
                                 == min(abs(Order_Book["Timestamp"] - Kline_Data["Timestamp"][0]))].to_list()[0]
        
        # Calculate the difference in time and drop the unnecessary rows
        diff = Order_Book["Timestamp"][index] - Kline_Data["Timestamp"][0]
        Order_Book = Order_Book.drop(Kline_Data.index[0:index]).reset_index(drop=True)
        
        # Match the times by substracting the difference
        Order_Book["Timestamp"] = Order_Book["Timestamp"] - diff
    
    # Removing duplicate rows just in case
    Order_Book = Order_Book[~Order_Book.duplicated('Timestamp', keep='first')]
    Kline_Data = Kline_Data[~Kline_Data.duplicated('Timestamp', keep='first')]
    
    # Converting timestamps from unix to datetime
    Order_Book.index = pd.to_timedelta(Order_Book["Timestamp"].rename("Time"), "ms")
    Order_Book.drop("Timestamp", axis=1, inplace=True)
    Kline_Data.index = pd.to_timedelta(Kline_Data["Timestamp"].rename("Time"), "ms")
    Kline_Data.drop("Timestamp", axis=1, inplace=True)
    
    # Upsampling to 250ms data incase of gaps
    Order_Book = Order_Book.resample(freq).ffill()
    Kline_Data = Kline_Data.resample(freq).ffill()
    
    # Returns joined dataset
    return Order_Book.join(Kline_Data)


def linear_data(Order_Book, Kline_data, l=5, d=20, N=1):
    """
    Build up linear data for linear model
    :param Order_Book: Raw Order Book Data (250ms ticks)
    :param Kline_Data: Raw Kline Data (250ms ticks)
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param d: the no. of DELAYS (in future) for calculating the mid price change
    :param N: a constant used while calculating MPB, refer the paper
    :return: Dataframe with the required metrics
    """
    
    # Joining raw data into a single dataframe
    data = join_data(Order_Book, Kline_data)
    
    # Specify naming convention
    convention = {
              "BestBid":"BidDiff", 
              "BidVol":"BVolDiff",
              "BestAsk": "AskDiff",
              "AskVol": "AVolDiff",
              "Turnover": "TurnDiff",
              "Volume": "VolDiff"
                }
    
    # Calculating first diferrences of various columns and dropping null rows
    ldata = data[["BestBid", "BidVol", "BestAsk", "AskVol", "Turnover", "Volume"]].diff().rename(columns=convention)
    ldata[["BidVol", "AskVol", "Price"]] = data[["BidVol", "AskVol", "Price"]]
    ldata["MidPrice"] = (data["BestAsk"] + data["BestBid"])/2
    
    # Dealing with a inverted market by straightening it up and weighting it, while weighting no spread as one tenth the tick size
    ldata["Spread"] = np.where(data["BestAsk"] - data["BestBid"] > 0, data["BestAsk"] - data["BestBid"], 
                               np.where(data["BestAsk"] - data["BestBid"] == 0, 0.01, 100))
    
    # Calculating Average of MidPrice for (t,t-1) to be used while calculating MPB
    ldata["AvgMP"] = (ldata["MidPrice"] + ldata["MidPrice"].shift(1))/2
    
    # Drop first column
    ldata.drop(ldata.index[0], inplace=True)
    
    # Calculating Mid Price Change for given delays
    ldata["MPC"] = ldata["MidPrice"].shift(-1).rolling(d).mean().shift(1-d) - ldata["MidPrice"]
    
    # Calculating Mid Price Basis
    ldata["MPB"] = np.where(ldata["VolDiff"] != 0, ((ldata.iloc[:,4]/ldata.iloc[:,5])/N), np.nan)
    ldata["MPB"] = ldata["MPB"].fillna(method='ffill')
    index_mpb = ldata.columns.get_loc("MPB")
    index_mp = ldata.columns.get_loc("MidPrice")
    ldata.iloc[0, index_mpb] = ldata.iloc[0, index_mp]
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
    ldata = ldata.drop(columns=ldata.columns[:8])
    ldata = ldata.drop(columns=["AvgMP"])
    
    # Return dataframe with required metrics
    return ldata.dropna()


if __name__ == '__main__':
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Build up linear data for linear model')

    # Add arguments
    parser.add_argument('order_book_path', type=str, help='Path to raw Order Book data CSV file')
    parser.add_argument('kline_data_path', type=str, help='Path to raw Kline Data CSV file')
    parser.add_argument('-l', type=int, default=5, help='The number of lags for VOI and OIR determined by the ACF Plot (default=5)')
    parser.add_argument('-d', type=int, default=20, help='The number of delays (in future) for calculating the mid price change (default=20)')
    parser.add_argument('-N', type=int, default=1, help='A constant used while calculating MPB, refer the paper (default=1)')

    # Parse the arguments
    args = parser.parse_args()
    
    # Read the data into a dataframe
    OrderBook = pd.read_csv(args.order_book_path)
    KlineData = pd.read_csv(args.kline_data_path)

    # Call the function with the parsed arguments
    data = linear_data(OrderBook, KlineData, args.l, args.d, args.N)
    print(data)
