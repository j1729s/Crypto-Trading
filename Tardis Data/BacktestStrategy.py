import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from BuildLinearModel import build_linear_model


class Action:
    """
    Executes trades based on strategy, keeps tracks of the position, and yields the trade information.
    """
    
    def __init__(self):
        # Initial State
        self.own = False
        self.position = 0
        self.TC = 0.000207
        self.cost = 0
        self.t_cost = 0
        self.t_volume = 0
        
    def open_trade(self, price):
        # Opens a position at 'day' start
        self.own = True
        self.position = 1 if price < 0 else -1
        self.t_cost += self.TC * abs(price)
        self.t_volume += 1
        self.cost = price
        
    def trade(self, price):
        # Trades by longing or shorting two contracts at once
        self.position = 1 if price < 0 else -1
        self.t_cost += 2 * self.TC * abs(price)
        self.t_volume += 2
        self.cost = 2* price
        
    def close_trade(self, price):
        # Closes the position at 'day' end
        self.position = 0
        self.t_cost += self.TC * abs(price)
        self.t_volume += 1
        self.cost = price
        self.own = False
        
    def __call__(self, data):
        # Parse through the data to identify the trade and yield the identifiers as a dictionary
        for index in data.index:
            
            signal, price = data.loc[index, "Signal"], data.loc[index, "Price"]
            
            # In position, BUY/SELL to:
            if self.own:
                
                # CLOSE   
                if index == data.index[-1]:
                    self.close_trade(price) if self.position == 1 else self.close_trade(-1*price)
                    yield {"Time": index, "Price": price, "Position": self.position, "Trade Cost": self.cost, 
                           "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost}
                    
                # TRADE
                elif self.position == 1 and signal == -1:
                    self.trade(price)
                    yield {"Time": index, "Price": price, "Position": self.position, "Trade Cost": self.cost, 
                           "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost} 
                elif self.position == -1 and signal == 1:
                    self.trade(-1*price)
                    yield {"Time": index, "Price": price, "Position": self.position, "Trade Cost": self.cost, 
                           "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost}
                      
            # NOT in position, BUY/SELL to OPEN
            else:
                self.open_trade(-1*price) if signal == 1 else self.open_trade(price)
                yield {"Time": index, "Price": price, "Position": self.position, "Trade Cost": self.cost, 
                           "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost}

                
def backtest_strategy(train_data, test_data, to_test='Pred', threshold=0.2, l=5, optimise=False):
    """
    Backtests the strategy and prints out the metrics
    :param train_data: Training Dataset
    :param test_data: Test data for prediction
    :param to_test: Backtest with real or predicted data
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param threshold: trading threshold
    :return: dataframe with Price, Predicted MPC, and True MPC as columns
    """
    
    if to_test == 'Pred':
        
        # Retrieve trained model
        model = build_linear_model(train_data, l=l)
    
        # Build the explanatory variables
        df = pd.DataFrame({'MPB': test_data["MPB"] / test_data["Spread"], 'VOI': test_data["VOI_(t)"] / test_data["Spread"], 
                       f'OIR': test_data["OIR_(t)"] / test_data["Spread"], **{f'VOI{i}': test_data[f"VOI_(t-{i})"] / test_data["Spread"] 
                        for i in range(1,l+1)}, **{f'OIR{i}': test_data[f"OIR_(t-{i})"] / test_data["Spread"] for i in range(1,l+1)}})
        
        # Predicting MPC and converting to multinomial classifier
        y_pred = model.predict(sm.add_constant(df))
        y_pred = pd.Series(np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0)), index=test_data.index)
        del df
        
        # Copying Data
        data = test_data[["Price"]].copy()
        
        # Formatting Data
        data["MPC_pred"] = y_pred
        data = data[["Price", "MPC_pred"]][(data['MPC_pred'] == 1) | (data['MPC_pred'] == -1)]
        
        # Return dataframe with Nan values in case there areno MPC in the given range for a threshold
        if len(data)==0:
            return pd.DataFrame(np.NaN, index=[0], columns=["Price","Position", "Trade Cost", "Volume", "Profit Before TC", "Transaction Cost", "Total Profit"])
        data.loc[test_data.index[-1]] = test_data.loc[test_data.index[-1], ["Price", "MPC"]]
        data.rename(columns = {"MPC_pred" : "Signal"}, inplace=True)
    
    elif to_test == 'Real':
        
        # Converting MPC to multinomial classifier
        y_true = pd.Series(np.where(test_data["MPC"] > threshold, 1, np.where(test_data["MPC"] < -threshold, -1, 0)), index=test_data.index)
        
        # Copying Data
        data = test_data[["Price"]].copy()
        
        # Formatting Data
        data["MPC"] = y_true
        data = data[["Price", "MPC"]][(data['MPC'] == 1) | (data['MPC'] == -1)]
        
        # Return dataframe with Nan values in case there areno MPC in the given range for a threshold
        if len(data)==0:
            return pd.DataFrame(np.NaN, index=[0], columns=["Price","Position", "Trade Cost", "Volume", "Profit Before TC", "Transaction Cost", "Total Profit"])
        data.loc[test_data.index[-1]] = test_data.loc[test_data.index[-1], ["Price", "MPC"]]
        data.rename(columns = {"MPC" : "Signal"}, inplace=True)
    
    # Create an instance of the class Action
    action = Action()
    
    # Fill in the data into a list of dictionaries
    return_list = []
    for trade in action(data):
        return_list.append(trade)
    
    # Convert list of dictionaries to a usable dataframe
    return_df = pd.DataFrame.from_dict(return_list)
    return_df.set_index("Time", inplace=True)
    
    # Add in the profit columns
    return_df["Profit Before TC"] = return_df["Profit Before TC"].cumsum()
    return_df["Total Profit"] = return_df["Profit Before TC"] - return_df["Transaction Cost"]
    
    # Print Metrics
    if optimise == False:
        
        print("Profit before transaction cost = {} USD".format(sum(return_df["Trade Cost"])))
        print("Transaction Cost = {} USD".format(return_df.iloc[-1,5]))
        print("Total Profit = {} USD".format(return_df.iloc[-1,6]))
        print("Total Trade Volume = {} trades".format(return_df.iloc[-1,3]))
        
    # Return Dataframe with Trade Data
    return return_df


if __name__ == '__main__':
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Backtest the trading strategy and print out the metrics.')
    
    # Add arguments
    parser.add_argument('train_file', help='Path to training dataset CSV file')
    parser.add_argument('test_file', help='Path to test dataset CSV file')
    parser.add_argument('--to_test', type=str, default='Pred', help='Backtest with real or predicted data')
    parser.add_argument('--threshold', type=float, default=0.2, help='Trading threshold (default=0.2)')
    parser.add_argument('--lags', type=int, default=5, help='The no. of LAGS for VOI and OIR determined by the ACF Plot (default=5)')
    parser.add_argument('--optimise', action='store_true', help='Enable model hyperparameter optimization')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the data into a dataframe
    train = pd.read_csv(args.train_file)
    test = pd.read_csv(args.test_file)
    
    # Call the function with the parsed arguments
    backtest_strategy(train, test, args.to_test, args.threshold, args.lags, args.optimise)