import dash
from dash import dcc, html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

TIME_INTERVAL = 500
C = 20
window = 300  # 600 * 500 = 300_000 i.e., a 5 minute window

def calculate_next_timestamp(timestamp):
    return (timestamp // TIME_INTERVAL + 1) * TIME_INTERVAL

def fair_value(file_path):
    df0 = pd.read_csv(file_path)
    start_time = calculate_next_timestamp(df0.iloc[0, 0])
    imbalance = []
    mid_price = []
    sum_bid = 0.0
    sum_ask = 0.0
    for index in df0.index:
        if df0.at[index, 'Time'] < start_time:
            sum_bid += df0.at[index, 'BidVol']
            sum_ask += df0.at[index, 'AskVol']
        elif df0.at[index, 'Time'] > start_time:
            start_time += TIME_INTERVAL
            mid_price.append((df0.at[index - 1, 'BestBid'] + df0.at[index - 1, 'BestAsk']) / 2)
            imbalance.append(sum_bid - sum_ask)
            sum_bid = 0
            sum_ask = 0
            sum_bid += df0.at[index, 'BidVol']
            sum_ask += df0.at[index, 'AskVol']
    alpha = []
    for t in range(len(imbalance)):
        m = np.nanmean(imbalance[max(0, t + 1 - window):t + 1])
        s = np.nanstd(imbalance[max(0, t + 1 - window):t + 1])
        alpha.append(np.divide(imbalance[t] - m, s))
    fair = [mid + C * alp for mid, alp in zip(mid_price, alpha)]
    x = np.arange(len(mid_price))

    # Create a subplot with two lines
    fig = make_subplots()

    # Plotting the first series (y1)
    trace1 = go.Scatter(x=x, y=mid_price, mode='lines+markers', name='Mid Price', line=dict(color='blue'))

    # Plotting the second series (y2)
    trace2 = go.Scatter(x=x, y=fair, mode='lines+markers', name='Fair', line=dict(color='red'))

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
    file_path = "okx_btcusdt_2023-12-05.csv"
    # Initialize Dash app
    app = dash.Dash(__name__)

    # Define the layout of the app
    app.layout = html.Div([
        dcc.Graph(figure=fair_value(file_path))
    ])

    app.run_server(port=8080, debug=True)