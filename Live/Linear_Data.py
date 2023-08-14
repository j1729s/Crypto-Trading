import numpy as np
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

def ffillna(arr):
    prev = np.arange(len(arr))
    prev[np.isnan(arr)] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

def shift(arr, n):
    if n >= 0:
        return np.concatenate((np.full(n, np.nan), arr[:-n]))
    else:
        return np.concatenate((arr[-n:], np.full(-n, np.nan)))

def linear_data_np(data, l=1, d=2, mpb='original'):
    data = np.array(data)
    
    # Calculate Turnover
    turnover = data[:, 4] * data[:, 5]
    
    # Calculate first differences
    ldata = np.diff(data, axis=0)
    
    # Calculate MidPrice
    mid_price = (data[1:, 2] + data[1:, 0]) / 2
    
    # Calculate AvgMP
    avg_mp = (mid_price + np.roll(mid_price, 1)) / 2
    
    # Calculate Spread
    spread = data[1:, 2] - data[1:, 0]
    
    # Calculate MPC
    mpc = np.convolve(np.roll(mid_price,-1)[:-1], np.ones(d), 'valid') / d - mid_price[:-d]
    mpc = np.concatenate((mpc, np.full(d, np.nan)))
    # Calculate MPB
    if mpb == 'updated':
        # Filling in values where volume changes and rest with Nan
        mpb = np.where(ldata[:, 5] == 0, np.nan, 
                       np.where(ldata[:, 4] != 0, (data[1:, 4] + (ldata[:, 5] / ldata[:, 4])), data[1:, 4]))
        # Filling in the iniial value
        mpb[0] = mid_price[0]
        # Filling in Nan
        mpb = ffillna(mpb)
        # Substracting average mid price
        mpb = mpb - avg_mp
    else:
        # Filling in values where volume changes and rest with Nan
        mpb = np.where(ldata[:, 5] == 0, np.nan, (np.diff(turnover) / ldata[:, 5]))
        # Filling in the iniial value
        mpb[0] = mid_price[0]
        # Filling in Nan
        mpb = ffillna(mpb)
        # Substracting average mid price
        mpb = mpb - avg_mp
        
    # Calculate OIR
    oir_t = (data[1:, 1] - data[1:, 3]) / (data[1:, 1] + data[1:, 3])
    
    # Calculate VOI
    dBid = np.where(ldata[:, 0] < 0, 0, 
                    np.where(ldata[:, 0] == 0, ldata[:, 1], data[1:, 1]))
    dAsk = np.where(ldata[:, 2] < 0, data[1:, 3], 
                    np.where(ldata[:, 2] == 0, ldata[:, 3], 0))
    voi_t = dBid - dAsk
    
    # Calculate OIR and VOI for each lag
    oir_lags = np.column_stack([shift(oir_t, i) for i in range(1, l+1)])
    voi_lags = np.column_stack([shift(voi_t, i) for i in range(1, l+1)])
    
    # Combine all calculated metrics
    result = np.column_stack((mpc, spread, mpb, oir_t, voi_t, oir_lags, voi_lags))
    
    # Remove rows with NaN values and return
    return result[~np.isnan(result).any(axis=1)]

def linear_model(train_data, l=1, d=2, mpb='updated'):
    """
    Build up linear model
    :param train_data: Training Dataset
    :param function: Determines what to be returned, the model or coefficient, default will return model
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :return: Linear model or coefficient
    """
    data = linear_data_np(train_data, l=l, d=d, mpb=mpb)
    # Build the linear model using OLS
    model = sm.OLS(data[:, 0], sm.add_constant((data[:, 2:].T/data[:, 1]).T)).fit()
    
    # Return Coefficients
    return model.params