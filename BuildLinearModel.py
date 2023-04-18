import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score


def build_linear_model(train_data, function='Validate', l=5):
    """
    Build up linear model
    :param train_data: Training Dataset
    :param function: Determines what to be returned, the model or coefficient, default will return model
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :return: Linear model or coefficient
    """
    
    # Build the explanatory variables
    df = pd.DataFrame({'y': train_data["MPC"], 'MPB': train_data["MPB"] / train_data["Spread"], 'VOI': train_data["VOI_(t)"] / train_data["Spread"], 
                       f'OIR': train_data["OIR_(t)"] / train_data["Spread"], **{f'VOI{i}': train_data[f"VOI_(t-{i})"] / train_data["Spread"] 
                        for i in range(1,l+1)}, **{f'OIR{i}': train_data[f"OIR_(t-{i})"] / train_data["Spread"] for i in range(1,l+1)}})
    
    # Scaling data?
    #scaler = MinMaxScaler(feature_range=(0.01, 10))
    #df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Build the linear model using OLS
    model = sm.OLS(df['y'], sm.add_constant(df.drop('y', axis=1))).fit()
    
    if function == 'Validate':
        # Return model
        return model
    else:
        # Return Coefficients
        return model.params


def validate_model(train_data, test_data, l=5, threshold=0.2):
    """
    Prints out metrics for model evaluation
    :param train_data: Training Dataset
    :param test_data: Test data for prediction
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param threshold: trading threshold
    """
    
    # Retrieve trained model
    model = build_linear_model(train_data, l=l)
    
    # Build the explanatory variables
    df = pd.DataFrame({'y': test_data["MPC"], 'MPB': test_data["MPB"] / test_data["Spread"], 'VOI': test_data["VOI_(t)"] / test_data["Spread"], 
                       f'OIR': test_data["OIR_(t)"] / test_data["Spread"], **{f'VOI{i}': test_data[f"VOI_(t-{i})"] / test_data["Spread"] 
                        for i in range(1,l+1)}, **{f'OIR{i}': test_data[f"OIR_(t-{i})"] / test_data["Spread"] for i in range(1,l+1)}})
    
    # Scaling data?
    #scaler = MinMaxScaler(feature_range=(0.01, 10))
    #df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    y_pred = model.predict(sm.add_constant(df.drop('y', axis=1)))
    y_true = df['y']
    
    # Calculate various metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    # Print the results
    print("Mean squared error: {:.2f}".format(mse))
    print("R^2 score: {:.2f}".format(r2))
    print("Explained variance score: {:.2f}".format(evs))
    
    # Converting to multinomial classifier
    y_true = np.where(df['y'] > threshold, 1, np.where(df['y'] < -threshold, -1, 0))
    y_pred = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    
    # calculate accuracy score for the predictions
    accuracy = accuracy_score(y_true, y_pred)

    # print the accuracy score
    print("Accuracy score: {:.2f}".format(accuracy))