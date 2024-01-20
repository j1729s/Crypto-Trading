import json
import logging
import asyncio
import websockets
import numpy as np
from time import sleep
from Okx_GLFT_model import config_model, get_model
from multiprocessing import Process, Condition, Array, Value
from Okx_GLFT_Helper import OrderPlacement, CancelOrders, CheckPosition

from Constants_GLFT import *
RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect


logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class DataHandler(Process):
    """Handles new data from websocket. Runs continuously"""
    def cal_params(self):
        print("New Params")
        lambda_bids = np.nanmean(self.bid_arr, axis=0)
        lambda_asks = np.nanmean(self.ask_arr, axis=0)
        self.bid_params[:] = get_model(lambda_bids)
        self.ask_params[:] = get_model(lambda_asks)

    def get_new_params(self):
        # Calculate model parameters
        bids, asks, candle = config_model('btcusdt')
        self.bid_arr, self.ask_arr = bids, asks
        self.candle_buffer[:] = candle[-VOL_TIME:]
        self.cal_params()

    def __init__(self, symbol, value, position_event, order_arr, order_event, cancel_event):
        Process.__init__(self)
        self.position = value
        self.position_event = position_event
        self.shared_orders = order_arr
        self.order = order_event
        self.cancel = cancel_event

        print("New Day")
        self.symbol = symbol

        ## Data Collection variables
        self.latest_book_ticker = None
        self.next_timestamp = 0
        self.last_timestamp = 0
        self.row = None

        ## Signal Generation variables
        self.count = 0
        self.interval_count = 0
        self.price = 0.0
        self.bid_spread = 0.0
        self.ask_spread = 0.0
        self.bid_skew = 0.0
        self.ask_skew = 0.0
        self.bid_params = np.full(2, np.nan)
        self.ask_params = np.full(2, np.nan)
        self.bid_buffer = np.full(41, np.nan)
        self.ask_buffer = np.full(41, np.nan)
        self.bid_arr = np.full((TRAINING_TIME, 41), np.nan)
        self.ask_arr = np.full((TRAINING_TIME, 41), np.nan)
        self.candle_buffer = np.full((VOL_TIME, 4), np.nan)
        self.second_buffer = np.full((1000 * round(TIME_INTERVAL / 1000), 3), np.nan)

        self.order_list = np.full(2, 0.0)
        self.tick_count = 0
        self.volatility = 0.0
        self.get_new_params()

    def calculate_next_timestamp(self, timestamp):
        return (timestamp // TIME_INTERVAL + 1) * TIME_INTERVAL

    async def send_orders(self):
        with self.order:
            self.order.notify()

    async def cancel_orders(self):
        with self.cancel:
            self.cancel.notify()

    async def order_cancel_wrapper(self):
        local_orders = np.frombuffer(self.shared_orders.get_obj(), dtype=np.float32)
        local_orders[:] = self.order_list
        await self.cancel_orders()
        await self.send_orders()

    @staticmethod
    def cal_mad(arr):
        arresh = (arr[:, 3] - arr[:, 0])
        return np.nanmean(np.abs(arresh - np.nanmean(arresh))) * np.sqrt(3.14 / 2)

    @staticmethod
    def remove_nans(count):
        # Identify rows with only NaN values
        nan_rows_mask = np.isnan(count).all(axis=1)
        # Filter out rows with only NaN values
        arr_without_nan = count[~nan_rows_mask, :]
        return arr_without_nan

    @staticmethod
    def compute_coeff(xi, gamma, delta, A, k, vol):
        inv_k = np.divide(1, k)
        c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
        c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
        half_spread = 1 * c1 + 1 / 2 * c2 * vol
        skew = c2 * vol
        return half_spread, skew

    def cal_intensity(self, tick):
        condition_bid = (self.second_buffer[:, 2] == -1) & (self.second_buffer[:, 0] <= np.round(self.price - tick, 1))
        condition_ask = (self.second_buffer[:, 2] == 1) & (self.second_buffer[:, 0] >= np.round(self.price + tick, 1))
        return np.nansum(self.second_buffer[:, 1][condition_bid]), np.nansum(self.second_buffer[:, 1][condition_ask])

    def add_new_interval(self):
        with self.position_event:
            self.position_event.notify()
        self.count = 0
        self.interval_count += 1
        for i in range(10, 51):
            self.bid_buffer[i - 10], self.ask_buffer[i - 10] = self.cal_intensity(i / 10)
        self.bid_arr = np.roll(self.bid_arr, -1, axis=0)
        self.ask_arr = np.roll(self.ask_arr, -1, axis=0)
        self.bid_arr[-1], self.ask_arr[-1] = self.bid_buffer, self.ask_buffer
        self.candle_buffer = np.roll(self.candle_buffer, -1, axis=0)
        second_buffer = self.remove_nans(self.second_buffer)
        self.candle_buffer[-1] = [second_buffer[:, 0][0], np.nanmin(second_buffer[:, 0]),
                                  np.nanmax(second_buffer[:, 0]), second_buffer[:, 0][-1]]
        self.volatility = self.cal_mad(self.candle_buffer)
        self.volatility = self.volatility if ~np.isnan(self.volatility) else 0.0
        print("Volatility is: ", self.volatility)
        self.second_buffer = np.full((1000 * round(TIME_INTERVAL / 1000), 3), np.nan)
        self.bid_spread, self.bid_skew = self.compute_coeff(GAMMA, GAMMA, DELTA, self.bid_params[0], self.bid_params[1], self.volatility)
        self.ask_spread, self.ask_skew = self.compute_coeff(GAMMA, GAMMA, DELTA, self.ask_params[0], self.ask_params[1], self.volatility)

    async def handle_new_message(self, current_event_time):
        """Puts the latest data into queue"""
        side = 1 if self.latest_book_ticker['side'] == 'buy' else -1
        if self.next_timestamp > current_event_time >= self.last_timestamp:
            if self.count < self.second_buffer.shape[0]:
                self.second_buffer[self.count] = np.array([self.latest_book_ticker['px'], self.latest_book_ticker['sz'], side], dtype=float)
                self.count += 1
        elif current_event_time >= self.next_timestamp:
            self.add_new_interval()
            self.last_timestamp = self.next_timestamp
            self.next_timestamp = self.calculate_next_timestamp(current_event_time)
            if self.next_timestamp > self.last_timestamp + TIME_INTERVAL:
                while self.last_timestamp < self.next_timestamp - TIME_INTERVAL:
                    self.second_buffer[0] = np.array([self.latest_book_ticker['px'], 0.0, 0], dtype=float)
                    self.last_timestamp += TIME_INTERVAL
                    self.add_new_interval()
            if self.interval_count >= TRAINING_INTERVAL:
                self.interval_count = 0
                self.cal_params()
            position = self.position.value
            print("Position is ", position)
            bid = np.minimum(self.row[0], np.round((self.price - self.bid_spread - self.bid_skew * position / int(CONTRACT_SIZE)), 1))
            ask = np.maximum(self.row[1], np.round((self.price + self.ask_spread - self.ask_skew * position / int(CONTRACT_SIZE)), 1))
            if position > Q:
                bid = 0
                ask = self.row[1]
            elif position < -1 * Q:
                bid = self.row[0]
                ask = 0
            self.order_list[:] = [bid, ask]
            print("Bid Spread and Skew are: ", self.bid_spread, self.bid_skew)
            print("Ask Spread and Skew are: ", self.ask_spread, self.ask_skew)
            await self.order_cancel_wrapper()
            self.second_buffer[self.count] = np.array([self.latest_book_ticker['px'], self.latest_book_ticker['sz'], side], dtype=float)
            self.price = float(self.latest_book_ticker['px'])
            self.count += 1


    async def on_message(self, data):
        event_data = data['data'][0]
        if data['arg']['channel'] == 'tickers':
            self.row = np.array([event_data["bidPx"], event_data["askPx"]], dtype=float)
        elif data['arg']['channel'] == 'trades':
            self.latest_book_ticker = event_data
            current_event_time = int(self.latest_book_ticker['ts'])
            if self.price == 0:
                self.price = float(self.latest_book_ticker['px'])
            if self.next_timestamp == 0:
                self.next_timestamp = self.calculate_next_timestamp(current_event_time)
            await self.handle_new_message(current_event_time)


    async def open_websocket(self):
        url = "wss://ws.okx.com:8443/ws/v5/public"
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    # Subscribe to the book ticker stream for the BTC-USDT trading pair
                    book_ticker_request = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "tickers",
                             "instId": self.symbol,
                             },
                            {"channel": "trades",
                             "instId": self.symbol,
                             }
                        ]
                    }
                    await websocket.send(json.dumps(book_ticker_request))
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)
                        if 'data' in data.keys():
                            await self.on_message(data)
            except Exception as e:
                logging.error(f"WebSocket encountered an error: {e}")
                logging.info(f"Reconnecting in {RECONNECT_DELAY} seconds...")
                sleep(RECONNECT_DELAY)

    async def data_collection(self):
        """Creates async task to handle data"""
        task = asyncio.create_task(self.open_websocket())
        await task

    def run(self):
        asyncio.run(self.data_collection())

def main():
    symbol = SYMBOL
    # Create a NumPy array
    original_array = np.full(2, 0.0)

    # Create a shared memory array
    value = Value('i', 0)
    ord_arr = Array('f', original_array)
    position_event = Condition()
    order_event = Condition()
    cancel_event = Condition()

    process1 = DataHandler(symbol, value, position_event, ord_arr, order_event, cancel_event)
    process1.start()
    # Start a process and pass the shared value
    process2 = CheckPosition(value, position_event)
    process2.start()

    process3 = OrderPlacement(ord_arr, order_event)
    process3.start()

    process4 = CancelOrders(ord_arr, cancel_event)
    process4.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()

if __name__ == "__main__":
    main()