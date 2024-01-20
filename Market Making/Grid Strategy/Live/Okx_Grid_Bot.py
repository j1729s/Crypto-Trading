import json
import logging
import asyncio
import websockets
import numpy as np
import pandas as pd
from time import sleep
from Okx_model_skew import linear_model
from Okx_Secret import API_KEY, API_SECRET, API_PASS
from datetime import datetime, timedelta
from multiprocessing import Process, Condition, Array, Value
from Okx_Order_Helper import OrderPlacement, CancelOrders, CheckPosition


from Constants import *
RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect
LAGS = 0
DELAYS = 1

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect

class DataHandler(Process):
    """Handles new data from websocket. Runs continuously"""

    def open_new_file(self):
        # Calculate model parameters
        try:
            data_date = datetime.utcnow().date() - timedelta(days=1)
            self.model = linear_model(f"okx_btcusdt_{data_date}.csv")
            print("New Model")
        except Exception as e:
            logging.error(f"Cannot open the file due to: {str(e)}")
            raise e  # re-throw the exception to be caught in the calling function

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
        self.row = None
        # Keeping track of new date
        self.date = datetime.utcnow().date()
        self.sum_bid = 0.0
        self.sum_ask = 0.0
        self.alpha = 0.0
        self.mid = 0.0
        self.ref_price = 0.0
        self.bid_price = 0.0
        self.ask_price = 0.0

        ## Signal Generation variables
        self.oi = 0.0
        self.model = None
        self.count = 0

        # Preallocate buffer with NaNs
        self.buffer = np.zeros(300)
        self.bid_list = np.arange(GRID_SIZE) * GRID_INTERVAL
        self.ask_list = np.arange(GRID_SIZE) * GRID_INTERVAL
        self.order_list = np.full(2 * GRID_SIZE + 2, 0.0)
        self.open_new_file()

    def calculate_next_timestamp(self, timestamp):
        return (timestamp // TIME_INTERVAL + 1) * TIME_INTERVAL

    def handle_new_date(self, new_date):
        self.date = new_date
        try:
            self.open_new_file()  # Open a new file
        except Exception as e:
            logging.error(f"Error opening new file: {e}")

    async def send_orders(self):
        with self.order:
            self.order.notify()

    async def cancel_orders(self):
        with self.cancel:
            self.cancel.notify()

    async def order_cancel_wrapper(self):
        local_orders = np.frombuffer(self.shared_orders.get_obj(), dtype=np.float32)
        local_orders[:] = self.order_list
        await self.send_orders()
        await self.cancel_orders()
    async def make_grid(self):
        position = self.position.value
        best_bid = self.row[1]
        best_ask = self.row[2]
        print(f"Position is {position}, Reference is {self.ref_price}")

        ## This part sets an upper and lower bound on the position size
        ## Which translates into posting 2 * GRID_SIZE orders on only one side of level 1 of LOB
        if position > round(MAX_RATIO * MAX_INVENTORY):
            print("Longing too much! Sending only sell orders now")
            self.order_list[1:-1] = np.full(2 * GRID_SIZE, best_ask)
            self.order_list[0] = 0
            self.order_list[-1] = -1
            await self.order_cancel_wrapper()
        elif position < -1 * round(MAX_RATIO * MAX_INVENTORY):
            print("Shorting too much! Sending only buy orders now")
            self.order_list[1:-1] = np.full(2 * GRID_SIZE, best_bid)
            self.order_list[0] = 0
            self.order_list[-1] = 1
            await self.order_cancel_wrapper()
        ## This part is when we are within the bounds of inventory
        else:
            self.bid_price = round(self.ref_price - HALF_SPREAD - SKEW * position, 1)
            self.ask_price = round(self.ref_price + HALF_SPREAD - SKEW * position, 1)

            self.bid_price = np.minimum(self.bid_price, best_bid)
            self.ask_price = np.maximum(self.ask_price, best_ask)
            self.bid_list = self.bid_price - self.bid_list
            self.ask_list = self.ask_price + self.ask_list

            ## This part posts at level 1 when position is higher or lower than +- MAX_INVENTORY
            ## The difference from above is that here we still post both sides
            if position > MAX_INVENTORY:
                self.ask_list = np.full(GRID_SIZE, best_ask)
            elif position < -1 * MAX_INVENTORY:
                self.bid_list = np.full(GRID_SIZE, best_bid)

            self.order_list[1:-1] = np.append(self.bid_list, self.ask_list)
            self.order_list[0] = 1
            self.order_list[-1] = -1
            await self.order_cancel_wrapper()

        self.bid_list = np.arange(GRID_SIZE) * GRID_INTERVAL
        self.ask_list = np.arange(GRID_SIZE) * GRID_INTERVAL


    async def generate_reference_price(self):
        if self.count < WINDOW - 1 and self.mid != 0:
            self.buffer[self.count] = self.oi
            self.count += 1
        elif self.count == WINDOW - 1:
            self.buffer[self.count] = self.oi
            m = np.nanmean(self.buffer)
            s = np.nanstd(self.buffer)
            self.alpha = np.divide(self.oi - m, s)
            self.ref_price = round(self.model.predict([self.alpha])[0] + self.mid, 1)
            #print(f"Last Mid Price: {round(self.mid, 2)}")
            self.buffer = np.roll(self.buffer, -1)
            await self.make_grid()


    async def handle_new_message(self, current_event_time):
        """Puts the latest data into queue"""
        if current_event_time > self.next_timestamp - 150 and current_event_time < self.next_timestamp:
            with self.position_event:
                self.position_event.notify()
        if current_event_time < self.next_timestamp:
            self.sum_bid += float(self.latest_book_ticker['bidSz'])
            self.sum_ask += float(self.latest_book_ticker['askSz'])
            self.row = np.array([self.next_timestamp, self.latest_book_ticker['bidPx'], self.latest_book_ticker['askPx']], dtype='float')
        elif current_event_time >= self.next_timestamp:
            self.oi = self.sum_bid - self.sum_ask
            await self.generate_reference_price()
            self.mid = (self.row[1] + self.row[2]) / 2
            self.row = np.array([self.next_timestamp, self.latest_book_ticker['bidPx'], self.latest_book_ticker['askPx']], dtype='float')
            self.next_timestamp = self.next_timestamp + TIME_INTERVAL
            self.sum_bid = 0.0
            self.sum_ask = 0.0
            self.sum_bid += float(self.latest_book_ticker['bidSz'])
            self.sum_ask += float(self.latest_book_ticker['askSz'])


    async def on_message(self, data):
        event_data = data['data'][0]
        self.latest_book_ticker = event_data

        new_date = datetime.utcnow().date()
        if new_date != self.date:
            self.handle_new_date(new_date)

        current_event_time = int(self.latest_book_ticker['ts'])
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
                             }]
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
    original_array = np.full(2 * GRID_SIZE + 2, 0.0)

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