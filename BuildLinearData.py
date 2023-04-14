import numpy as np
import pandas as pd

def join_data(Order_Book, Kline_Data):
    """
    Joins Raw data into usable dataframes
    :param Order_Book: Raw Order Book Data (250ms ticks)
    :param Kline_Data: Raw Kline Data (250ms ticks)
    :return: Joined dataframe with no gaps
    """
    
    # Removing unwanted column
    Order_Book = Order_Book.drop("Unnamed: 0", axis=1)
    Kline_Data = Kline_Data.drop("Unnamed: 0", axis=1)
    
    # Removing NaN values from Order Book Data
    Order_Book[["BestBid", "BestAsk", "MidPrice"]] = Order_Book[["BestBid", "BestAsk", "MidPrice"]].fillna(method='ffill')
    
    # Adding Turnover to Kline Data
    Kline_Data["Turnover"] = Kline_Data["NumberOfTrades"]*Kline_Data["Price"]*10 #Tick Value = Contract size * Tick size = $100 * $0.1
    
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
    Order_Book[~Order_Book.duplicated('Timestamp', keep='first')]
    Kline_Data[~Kline_Data.duplicated('Timestamp', keep='first')]
    
    # Converting timestamps from unix to datetime
    Order_Book.index = pd.to_timedelta(Order_Book["Timestamp"].rename("Time"), "ms")
    Order_Book.drop("Timestamp", axis=1, inplace=True)
    Kline_Data.index = pd.to_timedelta(Kline_Data["Timestamp"].rename("Time"), "ms")
    Kline_Data.drop("Timestamp", axis=1, inplace=True)
    
    # Upsampling to 250ms data incase of gaps
    Order_Book = Order_Book.resample("250ms").last().ffill()
    Kline_Data = Kline_Data.resample("250ms").last().ffill()
    
    # Returns joined dataset
    return Order_Book.join(Kline_Data)


def linear_data(Order_Book, Kline_data, l=5, d=20):
    """
    Build up linear data for linear model
    :param Order_Book: Raw Order Book Data (250ms ticks)
    :param Kline_Data: Raw Kline Data (250ms ticks)
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param d: the no. of DELAYS (in future) for calculating the mid price change
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
    ldata[["BidVol", "AskVol", "MidPrice", "Price"]] = data[["BidVol", "AskVol", "MidPrice", "Price"]]
    ldata.drop(ldata.index[0], inplace=True)
    
    # Calculating Mid Price Change for given delays
    ldata["MPC"] = ldata["MidPrice"].shift(-1).rolling(d).mean().shift(1-d) - ldata["MidPrice"]
    
    # Calculating Mid Price Basis
    ldata["MPB"] = np.where(ldata["VolDiff"] != 0, ((ldata.iloc[:,4]/ldata.iloc[:,5])/300), np.nan)
    ldata["MPB"] = ldata["MPB"].fillna(method='ffill')
    index_mpb = ldata.columns.get_loc("MPB")
    index_mp = ldata.columns.get_loc("MidPrice")
    ldata.iloc[0, index_mpb] = ldata.iloc[0, index_mp]
    
    # Calculating OIR
    ldata["OIR_(t)"] = (ldata["BidVol"] - ldata["AskVol"])/(ldata["BidVol"] + ldata["AskVol"])
    
    # Calculating VOI
    dBid = pd.Series(np.where(ldata["BidDiff"] < 0, 0, 
                              np.where(ldata["BidDiff"] == 0, ldata["BVolDiff"], ldata["BidVol"])), index=ldata.index)
    dAsk = pd.Series(np.where(ldata["AskDiff"] < 0, ldata["AskVol"], 
                              np.where(ldata["AskDiff"] == 0, ldata["AVolDiff"], 0)), index=ldata.index)
    ldata["VOI_(t)"] = dBid - dAsk
    
    # Calculating VOI for each lag by shifting data
    for i in range(1, l+1):
        ldata[f"OIR_(t-{i})"] = ldata["OIR_(t)"].shift(i)
        ldata[f"VOI_(t-{i})"] = ldata["VOI_(t)"].shift(i)
    # Dropping irrelevant columns
    ldata = ldata.drop(columns=ldata.columns[:8])
    
    # Return dataframe with required metrics
    return ldata.dropna()