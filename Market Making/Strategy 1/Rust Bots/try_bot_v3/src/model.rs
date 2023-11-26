use std::process::Command;
use std::io::{self, Write};
use std::str::from_utf8;
use rust_decimal::Decimal;
use crate::hyperparameters;

const SIZE: usize = hyperparameters::SIZE;

pub fn params_arr() -> [Decimal; SIZE] {
// Your Python code as a string
let python_code = r#"
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta

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
    assert l>=0, "Negative lags means forward projections"
    assert d>=1, "Delays have to be greater than 1"
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
    if l>0:
        # Calculate OIR and VOI for each lag
        oir_lags = np.column_stack([shift(oir_t, i) for i in range(1, l + 1)])
        voi_lags = np.column_stack([shift(voi_t, i) for i in range(1, l + 1)])

        # Combine all calculated metrics
        result = np.column_stack((mpc, spread, mpb, oir_t, voi_t, oir_lags, voi_lags))
    elif l == 0:
        result = np.column_stack((mpc, spread, mpb, oir_t, voi_t))

    
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

def open_new_file():
    # Calculate model parameters
    try:
        data_date = datetime.utcnow().date() - timedelta(days=1)
        params = linear_model(pd.read_csv(f"D:/Test/btcusdt_{data_date}.csv").drop('Time', axis=1), l=1, d=2)
        print(" ".join(map(str, params)))
    except Exception as e:
        print(f"Cannot open the file due to: {str(e)}")
        raise e  # re-throw the exception to be caught in the calling function
if __name__ == "__main__":
    open_new_file()
"#;

    // Create a temporary Python script file
    let mut script_file = std::fs::File::create("temp_script.py").expect("Failed to create Python script file");
    script_file.write_all(python_code.as_bytes()).expect("Failed to write Python code to the script file");

    // Run the Python script using the Python interpreter
    let output = Command::new("python")
        .arg("temp_script.py")
        .output()
        .expect("Failed to execute Python script");

    // Clean up the temporary script file
    std::fs::remove_file("temp_script.py").expect("Failed to remove the temporary script file");

    // Check if the Python script execution was successful
    if output.status.success() {
        // Parse the Python script's output as a UTF-8 string
        let stdout_str = from_utf8(&output.stdout).expect("Failed to parse stdout as UTF-8");
        
        // Extract the array part (without brackets and newline characters)
        // Split the cleaned string into parts using ' ' as the delimiter
        let arr_str: Vec<&str> = stdout_str.trim().split(' ').collect();

        // Parse the string parts as f64 values (We need this because rust can't parse scientific notation to decimal)  
        let parsed_values: Vec<f64> = arr_str.iter().map(|&s| s.parse::<f64>().unwrap()).collect();
        
        // Parse the array of string into a Rust vector of Decimal values
        let mut returned_values = [Decimal::default(); SIZE];
        for (i, value) in parsed_values.iter().enumerate() {
            returned_values[i] = Decimal::from_f64_retain(*value).unwrap_or_default();
        }
        
        returned_values
        
    } else {
        // Display any errors from the Python script
        io::stderr().write_all(&output.stderr).unwrap();

        // Return an empty vector in case of an error
        [Decimal::default(); SIZE]
    }
}