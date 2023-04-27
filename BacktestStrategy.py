import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from BuildLinearModel import build_linear_model
warnings.filterwarnings("ignore")


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
        
        # Predicting MPC
        y_pred = model.predict(sm.add_constant(df))
    
        # Converting to multinomial classifier
        y_pred = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
        y_true = pd.Series(np.where(test_data["MPC"] > threshold, 1, np.where(test_data["MPC"] < -threshold, -1, 0)), index=test_data.index)
        test_data["MPC_pred"] = y_pred
        
        df = test_data[["Price", "MPC_pred"]]
        df["MPC"] = y_true
        data = test_data['MPC_pred']
    
    elif to_test == 'Real':
        
        y_true = pd.Series(np.where(test_data["MPC"] > threshold, 1, np.where(test_data["MPC"] < -threshold, -1, 0)), index=test_data.index)
        df = test_data[["Price"]]
        data = y_true
    
    # Define Constants
    own = False
    position = 0
    TC = 0.000207
    cost = []
    t_cost = 0
    t_volume = 0
    return_df = pd.DataFrame(columns=["Time", "Price", "Position", "Profit"])
    
    for index in data.index:
        
        # BUY to OPEN
        if own == False and data.loc[index] == 1:
            
            own = True
            position = 1
            price = df.loc[index, "Price"]
            cost.append(-1*price)
            t_cost += TC*price
            t_volume += 1
            return_df.loc[len(return_df)] = [index, price, position, (sum(cost) - t_cost)]
            
        # SELL to OPEN    
        elif own == False and data.loc[index] == -1:
           
            own = True
            position = -1
            price = df.loc[index, "Price"]
            cost.append(price)
            t_cost += TC*price
            t_volume += 1
            return_df.loc[len(return_df)] = [index, price, position, (sum(cost) - t_cost)]
            
        # SELL to TRADE
        elif own and position == 1 and data.loc[index] == -1:
            
            own == True
            position = -1
            price = df.loc[index, "Price"]
            cost.append(price)
            cost.append(price)
            t_cost += 2*TC*price
            t_volume += 2
            return_df.loc[len(return_df)] = [index, price, position, (sum(cost) - t_cost)]
            
        # BUY to TRADE    
        elif own and position == -1 and data.loc[index] == 1:
            
            own == True
            position = 1
            price = df.loc[index, "Price"]
            cost.append(-1*price)
            cost.append(-1*price)
            t_cost += 2*TC*price
            t_volume += 2
            return_df.loc[len(return_df)] = [index, price, position, (sum(cost) - t_cost)]
            
        # CLOSE at day end
        elif position == 1 and index == data.index[-1]:
            
            position = 0
            price = df.loc[index, "Price"]
            cost.append(price)
            t_cost += TC*price
            t_volume += 1
            return_df.loc[len(return_df)] = [index, price, position, (sum(cost) - t_cost)]
            
        # CLOSE at day end
        elif position == -1 and index == data.index[-1]:
            
            position = 0
            price = df.loc[index, "Price"]
            cost.append(-1*price)
            t_cost += TC*price
            t_volume += 1
            return_df.loc[len(return_df)] = [index, price, position, (sum(cost) - t_cost)]
            
    # Print Metrics
    if optimise == False:
        
        print("Profit before transaction cost = {} USD".format(sum(cost)))
        print("Transaction Cost = {} USD".format(t_cost))
        print("Total Profit = {} USD".format(sum(cost)-t_cost))
        print("Total Trade Volume = {} trades".format(t_volume))
    
        # Return Trade data
        return_df.set_index("Time", inplace=True)
        return return_df.join(df[["MPC","MPC_pred"]])
    
    # Use this when optimising
    if optimise:
        
        # Return trading cost, transaction cost and trade volume data
        return cost, t_cost, t_volume


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
