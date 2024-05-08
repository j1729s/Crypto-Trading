import dash
from dash import dcc, html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import statsmodels.api as sm

TIME_INTERVAL = 1000
window = 300  # 600 * 500 = 300_000 i.e., a 5 minute window

def calculate_next_timestamp(timestamp):
    return (timestamp // TIME_INTERVAL + 1) * TIME_INTERVAL

def calculate_predictions(df0):
    start_time = calculate_next_timestamp(df0.iloc[0, 0])
    imbalance = []
    mid_price = []
    vi = []
    sum_bid = 0.0
    sum_ask = 0.0
    sum_bid_last = None
    sum_ask_last = None
    for index in df0.index:
        if df0.at[index, 'Time'] < start_time:
            sum_bid += df0.at[index, 'BidVol']
            sum_ask += df0.at[index, 'AskVol']
        elif df0.at[index, 'Time'] > start_time:
            start_time += TIME_INTERVAL
            best_bid = df0.at[index - 1, 'BestBid']
            best_ask = df0.at[index - 1, 'BestAsk']
            if sum_bid_last is None:
                sum_bid_last = sum_bid
                sum_ask_last = sum_ask
            else:
                mid_price.append((best_bid + best_ask) / 2)
                imbalance.append((sum_bid - sum_bid_last) - (sum_ask - sum_ask_last))
                vi.append(sum_bid / (sum_bid + sum_ask))
                sum_bid_last = sum_bid
                sum_ask_last = sum_ask
            sum_bid = 0
            sum_ask = 0
            sum_bid += df0.at[index, 'BidVol']
            sum_ask += df0.at[index, 'AskVol']
    alpha = []
    for t in range(window, len(imbalance)):
        m = np.nanmean(imbalance[t - window:t])
        s = np.nanstd(imbalance[t - window:t])
        alpha.append(np.divide(imbalance[t] - m, s))
    # Calculating the change in midprice between this and previous interval
    midesh = np.array(mid_price)[window:] - np.roll(mid_price, 1)[window:]
    alphesh = np.roll(imbalance[window:], 1)
    q1 = np.percentile(alphesh, 10)
    q3 = np.percentile(alphesh, 90)
    return midesh, alphesh, mid_price, q1, q3, vi

def fair_value(file_path_train, file_path_test):
    df0 = pd.read_csv(file_path_train)
    midesh, alphesh, _, q1, q3, _ = calculate_predictions(df0)

    df = pd.read_csv(file_path_test)
    # Create a linear regression model
    model = sm.OLS(midesh, alphesh)

    # Fit the model to the data
    results = model.fit()
    midesh, alphesh, mid_price, _, _, vi = calculate_predictions(df)
    predictions = results.predict(alphesh) + np.roll(mid_price, 1)[window:]

    x = np.arange(len(mid_price[window:]))

    # Create a subplot with two lines
    fig = make_subplots()
    #colors = np.where(alphesh > q3, 'green', np.where(alphesh < q1, 'red', 'white'))

    # Plotting the first series (y1)
    trace1 = go.Scatter(x=x, y=mid_price[window:], mode='lines+markers', name='Mid Price', line=dict(color='blue'))

    # Plotting the second series (y2)
    trace2 = go.Scatter(x=x, y=np.array(vi[window:])+np.array(mid_price[window:]), mode='markers', name='VI', marker=dict(symbol='circle', size=5, opacity=0.9, color=vi[window:], colorscale='RdYlGn'))

    # Add traces to the subplot
    fig.add_trace(trace1)
    fig.add_trace(trace2)

    # Update layout with labels and title
    fig.update_layout(
        xaxis=dict(title='X-axis Label'),
        yaxis=dict(title='Y-axis Label'),
        title='Two Series Plot',
        showlegend=True)
    return fig

# Run the app
if __name__ == '__main__':
    file_path_train = "okx_btcusdt_2023-12-04.csv"
    file_path_test = "okx_btcusdt_2023-12-05.csv"
    # Initialize Dash app
    app = dash.Dash(__name__)

    # Define the layout of the app
    app.layout = html.Div([
        dcc.Graph(figure=fair_value(file_path_train, file_path_test))
    ])

    app.run_server(port=8050, debug=True)