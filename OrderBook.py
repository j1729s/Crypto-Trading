import websocket
import json
import pandas as pd
from Secrets import api_key, api_secret

def get_order_book(ticker):
    """
    Gets relevant 250ms Kline data snapshots for the ticker
    :param ticker: ticker
    :return: Dataframe with the metrics
    """

    symbol = ticker.lower()

    data = pd.DataFrame(columns=['Timestamp', 'BestBid', 'BestAsk', 'MidPrice', 'AskVol', 'BidVol'])

    # create the WebSocket connection and provide the API key in the headers

    ws = websocket.WebSocket()

    ws.connect(f"wss://fstream.binance.com/ws/{symbol}@depth",
               header={"X-MBX-APIKEY": f"{api_key}"})

    # subscribe to the desired WebSocket stream

    subscribe_message = json.dumps({
        "method": "SUBSCRIBE",
        "params":
            [
                f"{symbol}@depth<5>_250ms"
            ],
        "id": 1
    })

    ws.send(subscribe_message)

    # receive messages from the WebSocket stream
    try:
        while True:
            result = json.loads(ws.recv())
            if len(result) > 2:
                data = on_message_order(result, data)
    except KeyboardInterrupt:
        return data


def on_message_order(results, data):
    """
    For parsing through the request result
    :param results: the request result
    :param data: dataframe with previous data
    :return: dataframe with another row filled in
    """
    frames = {side: pd.DataFrame(data=results[side], columns=["price", "quantity"],
                                 dtype=float) for side in ["b", "a"]}

    tick_size = 0.1

    best_bid = max(frames['b']['price'])
    best_ask = min(frames['a']['price'])
    mid_price = (best_bid + best_ask) / 2

    ask_vol = frames['a'].quantity[frames['a'].price < mid_price + tick_size].sum()
    bid_vol = frames['b'].quantity[frames['b'].price > mid_price - tick_size].sum()

    data.loc[len(data)] = [results['E'], best_bid, best_ask, mid_price, ask_vol, bid_vol]
    return data