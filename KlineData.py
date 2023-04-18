import websocket
import json
import pandas as pd
from Secrets import api_key, api_secret

def get_kline(ticker):
    """
    Gets relevant 250ms Kline data snapshots for the ticker
    :param ticker: ticker
    :return: Dataframe with the metrics
    """

    symbol = ticker.lower()

    data = pd.DataFrame(columns=['Timestamp', 'Price', 'Volume', 'NumberOfTrades'])

    # create the WebSocket connection and provide the API key in the headers
    ws = websocket.WebSocket()

    ws.connect(f"wss://fstream.binance.com/ws/{symbol}@kline",
               header={"X-MBX-APIKEY": f"{api_key}"})

    # subscribe to the desired WebSocket stream
    subscribe_message = json.dumps({
        "method": "SUBSCRIBE",
        "params":
            [
                f"{symbol}@kline_1m"
            ],
        "id": 1
    })

    ws.send(subscribe_message)

    # receive messages from the WebSocket stream
    try:
        while True:
            result = json.loads(ws.recv())
            if len(result) > 2:
                data = on_message_kline(result, data)
    except KeyboardInterrupt:
        print(data)
        return data


def on_message_kline(results, data):
    """
    For parsing through the request result
    :param results: the request result
    :param data: dataframe with previous data
    :return: dataframe with another row filled in
    """

    price = results['k']['c']
    volume = results['k']['v']
    trade_no = results['k']['n']

    data.loc[len(data)] = [results['E'], price, volume, trade_no]
    return data

if __name__ == '__main__':
    KlineData = get_kline('BTCUSDT')
