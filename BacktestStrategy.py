import numpy as np
import pandas as pd
import statsmodels.api as sm
from BuildLinearModel import build_linear_model


def backtest_strategy(train_data, test_data, threshold=0.2, l=5):
    """
    Backtests the strategy and prints out the metrics
    :param train_data: Training Dataset
    :param test_data: Test data for prediction
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param threshold: trading threshold
    :return: dataframe with Price, Predicted MPC, and True MPC as columns
    """
    
    # Retrieve trained model
    model = build_linear_model(train_data, l=l)
    
    # Build the explanatory variables
    df = pd.DataFrame({'MPB': test_data["MPB"] / test_data["Spread"], 'VOI': test_data["VOI_(t)"] / test_data["Spread"], 
                       f'OIR': test_data["OIR_(t)"] / test_data["Spread"], **{f'VOI{i}': test_data[f"VOI_(t-{i})"] / test_data["Spread"] 
                        for i in range(1,l+1)}, **{f'OIR{i}': test_data[f"OIR_(t-{i})"] / test_data["Spread"] for i in range(1,l+1)}})
    
    y_pred = model.predict(sm.add_constant(df))
    y_pred = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    test_data["MPC_pred"] = y_pred
    y_true = np.where(test_data["MPC"] > threshold, 1, np.where(test_data["MPC"] < -threshold, -1, 0))
    df = test_data[["Price", "MPC_pred"]]
    df["MPC"] = y_true
    
    own = False
    position = 0
    TC = 0.0003
    cost = []
    t_cost = 0
    t_volume = 0
    
    for index in df.index:
        
        # BUY to OPEN
        if own == False and df.loc[index, "MPC_pred"] == 1:
            
            own = True
            position = 1
            price = df.loc[index, "Price"]
            cost.append(-1*price)
            t_cost += TC*price
            t_volume += 1
            
        # SELL to OPEN    
        elif own == False and df.loc[index, "MPC_pred"] == -1:
           
            own = True
            position = -1
            price = df.loc[index, "Price"]
            cost.append(price)
            t_cost += TC*price
            t_volume += 1
            
        # SELL to TRADE
        elif own and position == 1 and df.loc[index, "MPC_pred"] == -1:
            
            own == True
            position = -1
            price = df.loc[index, "Price"]
            cost.append(price)
            cost.append(price)
            t_cost += 2*TC*price
            t_volume += 2
            
        # BUY to TRADE    
        elif own and position == -1 and df.loc[index, "MPC_pred"] == 1:
            
            own == True
            position = 1
            price = df.loc[index, "Price"]
            cost.append(-1*price)
            cost.append(-1*price)
            t_cost += 2*TC*price
            t_volume += 2
        
        # CLOSE at day end
        elif position == 1 and index == df.index[-1]:
            
            price = df.loc[index, "Price"]
            cost.append(price)
            t_cost += TC*price
            t_volume += 1
        
        # CLOSE at day end
        elif position == -1 and index == df.index[-1]:
            
            price = df.loc[index, "Price"]
            cost.append(-1*price)
            t_cost += TC*price
            t_volume += 1
            
    print("Total Profit = {} USD".format(sum(cost)-t_cost))
    print("Total Trade Volume = {} trades".format(t_volume))
    
    return df