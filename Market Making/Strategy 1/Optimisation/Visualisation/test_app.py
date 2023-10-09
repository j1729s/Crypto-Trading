import dash
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objs as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from BuildLinearModel import build_linear_model
from Linear_Data_pd import linear_data

# Create an empty list to store plot figures
plot_figs = []
num_quantiles = 5

def backtest_strategy(train_data, test_data, l=0):
    """
    Backtests the strategy and prints out the metrics
    :param train_data: Training Dataset
    :param test_data: Test data for prediction
    :param to_test: Backtest with real or predicted data
    :param l: the no. of LAGS for VOI and OIR determined by the ACF Plot
    :param threshold: trading threshold
    :return: dataframe with Price, Predicted MPC, and True MPC as columns
    """

    # Retrieve trained model
    model = build_linear_model(train_data, l=l)

    # Build the explanatory variables
    if l > 0:
        df = pd.DataFrame(
            {'MPB': test_data["MPB"] / test_data["Spread"], 'VOI': test_data["VOI_(t)"] / test_data["Spread"],
             f'OIR': test_data["OIR_(t)"] / test_data["Spread"],
             **{f'VOI{i}': test_data[f"VOI_(t-{i})"] / test_data["Spread"]
                for i in range(1, l + 1)},
             **{f'OIR{i}': test_data[f"OIR_(t-{i})"] / test_data["Spread"] for i in range(1, l + 1)}})
    elif l == 0:
        df = pd.DataFrame(
            {'MPB': test_data["MPB"] / test_data["Spread"], 'VOI': test_data["VOI_(t)"] / test_data["Spread"],
             f'OIR': test_data["OIR_(t)"] / test_data["Spread"]})

    # Predicting MPC and converting to multinomial classifier
    y_pred = model.predict(sm.add_constant(df))
    del df

    # Copying Data
    data = test_data[["Time", "Price"]].copy()

    # Formatting Data
    data["MPC_pred"] = y_pred
    data = data[["Time", "Price", "MPC_pred"]]
    return data

def make_subplots(df):
    global plot_figs

    df['quantile'] = pd.qcut(df['Time'], num_quantiles, labels=False)
    df['Time'] = pd.to_datetime(df['Time'], unit="ms").dt.strftime('%H:%M:%S.%f')

    # Create plots for each quantile
    for i in range(num_quantiles):
        quantile_df = df[df['quantile'] == i]

        line_trace = go.Scatter(
            x=quantile_df['Time'],
            y=quantile_df['Price'],
            mode='lines',
            name=f'Quantile {i + 1} (Line)',
        )

        quantile_df = quantile_df[(quantile_df['MPC_pred'] > 0.1) | (quantile_df['MPC_pred'] < -0.1)]

        scatter_trace = go.Scatter(
            x=quantile_df['Time'],
            y=quantile_df['Price'],
            mode='markers',
            marker=dict(symbol='circle', size=5, opacity=0.9, color=quantile_df['MPC_pred'], colorscale='RdYlGn'),
            name=f'Quantile {i + 1} (Scatter)',
        )

        # Create a subplot with both traces
        subplot = go.Figure(data=[line_trace, scatter_trace])

        # Add the subplot to the list of plot figures
        plot_figs.append(subplot)


# Initialize the Dash app
app = dash.Dash(__name__)

# Define app layout with tabs for each quantile's plot
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label=f'Quantile {i + 1}', value=f'tab-{i + 1}') for i in range(num_quantiles)
    ]),
    html.Div(id='tabs-content'),
])


# Define callback to update tab content
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    return dcc.Graph(figure=plot_figs[int(tab.split('-')[1]) - 1])


if __name__ == '__main__':

    # Read the data into a dataframe
    train_data = linear_data(pd.read_csv("btcusdt_2023-09-03.csv"), 0, 1)
    test_data = linear_data(pd.read_csv("btcusdt_2023-09-04.csv"), 0, 1)

    # Call the function with the parsed arguments
    data = backtest_strategy(train_data, test_data, l=0)
    make_subplots(data)
    app.run_server(debug=True)