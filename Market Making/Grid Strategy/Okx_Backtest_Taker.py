import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from Okx_Process_Data import linear_model


class Action:
    """
    Executes trades based on strategy, keeps tracks of the position, and yields the trade information.
    """

    def __init__(self, size):
        # Initial State
        self.own = False
        self.position = 0
        self.TC = 0.000
        self.cost = 0
        self.t_cost = 0
        self.t_volume = 0
        self.size = size
        self.contract_size = 0.01

    def open_trade(self, price):
        # Opens a position at 'day' start
        self.own = True
        self.position = 1 if price < 0 else -1
        self.t_cost += self.size * self.TC * abs(price)
        self.t_volume += 1 * self.size
        self.cost = price * self.size * self.contract_size

    def trade(self, price):
        # Trades by longing or shorting two contracts at once
        self.position = 1 if price < 0 else -1
        self.t_cost += 2 * self.TC * abs(price) * self.size
        self.t_volume += 2 * self.size
        self.cost = 2 * price * self.size * self.contract_size

    def close_trade(self, price):
        # Closes the position at 'day' end
        self.position = 0
        self.t_cost += self.size * self.TC * abs(price)
        self.t_volume += 1 * self.size
        self.cost = price * self.size * self.contract_size
        self.own = False

    def __call__(self, data):
        # Parse through the data to identify the trade and yield the identifiers as a dictionary
        for index in data.index:

            signal, buy, sell = data.loc[index, "Signal"], data.loc[index, "BestBid"], data.loc[index, "BestAsk"]

            # In position, BUY/SELL to:
            if self.own:

                # CLOSE
                if index == data.index[-1]:
                    price = sell if self.position == 1 else -1 * buy
                    self.close_trade(price)
                    yield {"Time": index, "Price": price, "Position": self.position, "Trade Cost": self.cost,
                           "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost}

                # TRADE
                elif self.position == 1 and signal == -1:
                    self.trade(sell)
                    yield {"Time": index, "Price": sell, "Position": self.position, "Trade Cost": self.cost,
                           "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost}
                elif self.position == -1 and signal == 1:
                    self.trade(-1 * buy)
                    yield {"Time": index, "Price": -1 * buy, "Position": self.position, "Trade Cost": self.cost,
                           "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost}

            # NOT in position, BUY/SELL to OPEN
            else:
                price = -1 * buy if signal == 1 else sell
                self.open_trade(price)
                yield {"Time": index, "Price": price, "Position": self.position, "Trade Cost": self.cost,
                       "Volume": self.t_volume, "Profit Before TC": self.cost, "Transaction Cost": self.t_cost}


def backtest_taker(train_data, test_data, threshold=0.2, l=0, optimise=False, position_size=1):
    assert position_size >= 1, "Balance too LOW to take a position!!"
    """
    Backtest the strategy and prints out the metrics
    :param train_data: Training Dataset
    :param test_data: Test data for prediction
    :param l: the no. of LAGS for OI
    :param threshold: trading threshold
    :return: dataframe with Price, Predicted MPC, and True MPC as columns
    """

    # Retrieve trained model
    model = linear_model(train_data, l=l)

    df = pd.DataFrame({'OI': test_data["OI_(t)"] / test_data['Spread'],
                       **{f'OI{i}': test_data[f"OI_(t-{i})"] / test_data['Spread'] for i in range(1, l + 1)}})

    # Predicting MPC and converting to multinomial classifier
    y_pred_actual = model.predict(sm.add_constant(df))
    y_pred = pd.Series(np.where(y_pred_actual > threshold, 1, np.where(y_pred_actual < -threshold, -1, 0)),
                       index=test_data.index)
    del df

    # Copying Data
    data = test_data[["BestBid", "BestAsk"]].copy()

    # Formatting Data
    data["MPC_pred_actual"] = y_pred_actual
    data["MPC_pred"] = y_pred
    data = data[(data['MPC_pred'] == 1) | (data['MPC_pred'] == -1)]

    # Return dataframe with Nan values in case there are no MPC in the given range for a threshold
    if len(data) == 0:
        return pd.DataFrame(np.NaN, index=[0], columns=["Price", "Position", "Trade Cost", "Volume", "Profit Before TC",
                                                        "Transaction Cost", "Total Profit"])
    data.loc[test_data.index[-1]] = test_data.loc[test_data.index[-1], ["BestBid", "BestAsk", "MPC"]]
    data.rename(columns={"MPC_pred": "Signal"}, inplace=True)

    # Create an instance of the class Action
    action = Action(position_size)

    # Fill in the data into a list of dictionaries
    return_list = []
    for trade in action(data):
        return_list.append(trade)

    # Convert list of dictionaries to a usable dataframe
    return_df = pd.DataFrame.from_dict(return_list)

    # Merge DataFrames based on the common column
    return_df.set_index("Time", inplace=True)

    # Add in the profit columns
    return_df["Profit Before TC"] = return_df["Profit Before TC"].cumsum()
    return_df["Total Profit"] = return_df["Profit Before TC"] - return_df["Transaction Cost"]

    # Print Metrics
    if optimise == False:
        print("Profit before transaction cost = {} USD".format(sum(return_df["Trade Cost"])))
        print("Transaction Cost = {} USD".format(return_df.iloc[-1, 5]))
        print("Total Profit = {} USD".format(return_df.iloc[-1, 6]))
        print("Total Trade Volume = {} trades".format(return_df.iloc[-1, 3]))

    # Return Dataframe with Trade Data
    return return_df


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Backtest the trading strategy and print out the metrics.')

    # Add arguments
    parser.add_argument('train_file', help='Path to training dataset CSV file')
    parser.add_argument('test_file', help='Path to test dataset CSV file')
    parser.add_argument('--threshold', type=float, default=0.2, help='Trading threshold (default=0.2)')
    parser.add_argument('--lags', type=int, default=5,
                        help='The no. of LAGS for VOI and OIR determined by the ACF Plot (default=5)')
    parser.add_argument('--optimise', action='store_true', help='Enable model hyperparameter optimization')
    parser.add_argument('--position_size', type=int, default=1, help='Position size (default: 1)')

    # Parse the arguments
    args = parser.parse_args()

    # Read the data into a dataframe
    train = pd.read_csv(args.train_file)
    test = pd.read_csv(args.test_file)

    # Call the function with the parsed arguments
    backtest_taker(train, test, args.threshold, args.lags, args.optimise, args.position_size)