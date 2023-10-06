import time
import ccxt
import asyncio
import websockets
import orjson
import logging
import schedule
import numpy as np
import pandas as pd
from time import sleep
from Linear_Data import linear_model
from decimal import Decimal, getcontext
from Secrets import API_KEY, API_SECRET
from datetime import datetime, timedelta
from threading import Lock, Thread, Condition
from queue import LifoQueue

getcontext().prec = 11
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect
LEVERAGE = 125
CONTRACT_SIZE = 0.001
CHECK_LAG = 2  # In seconds

# Define Global Variables
alpha = [None, None]
alpha_lock = Lock()
# Event (boolean operator) used to fire process by setting or clearing
signal_event = Condition()

class CheckOrder_Panic(Thread):
    OPEN_ORDER_THRESHOLD = 25
    CANCEL_ORDER_THRESOLD = 5

    def __init__(self, symbol):
        Thread.__init__(self)

        self.symbol = symbol
        self.runtime = 0

        ## Exchange info and constants
        self.binance = ccxt.binanceusdm({'apiKey': API_KEY,
                                         'secret': API_SECRET})
        self.binance.load_markets()  # load markets to get the market id from a unified symbol

    def check_position(self):
        start = time.time()
        print("Scheduled Check")
        open_orders = self.binance.fetchOpenOrders(self.symbol)
        len_order = len(open_orders)
        print(f"Open Orders: {len_order}")
        if len_order > self.OPEN_ORDER_THRESHOLD:
            order_ids = [open_orders[i]["id"] for i in range(self.CANCEL_ORDER_THRESOLD)]
            print("Cancelling")
            for id in order_ids:
                try:
                    self.binance.cancelOrder(id, symbol=self.symbol, params={})
                except Exception as e:
                    print("Error while cancelling: {}", e)
        end = time.time()
        self.runtime = round(end - start)

    def run(self):
        schedule.every(CHECK_LAG).seconds.do(self.check_position)
        while True:
            schedule.run_pending()
            sleeptime = (CHECK_LAG - self.runtime) if self.runtime < CHECK_LAG else 0
            sleep(sleeptime)

class OrderPlacement(Thread):
    """Concerned with order placement. Fires when signal_event is set"""

    def __init__(self, symbol, price_row, price_lock):
        Thread.__init__(self)
        ## Multithreading variables
        self.price_lock = price_lock
        self.price_row = price_row

        ## Trading Metrics
        self.alpha = None
        self.symbol = symbol
        self.signal = 0
        self.size = CONTRACT_SIZE
        self.price = 0.0

        ## Exchange info and constants
        self.binance = ccxt.binanceusdm({'apiKey': API_KEY,
                                         'secret': API_SECRET})
        self.binance.load_markets()  # load markets to get the market id from a unified symbol
        self.binance.setLeverage(LEVERAGE, symbol=self.symbol)  # set account cross leverage


    def place_order(self, side):
        """Places orders. Checks if the alpha and price correspond to the latest alpha and price"""
        global alpha, alpha_lock
        try:
            # Checking for alpha and price correspondence
            with self.price_lock:
                last_row = self.price_row.get_nowait()
            with alpha_lock:
                alpha = [None, None]
            cost = last_row[0] if side == 'buy' else last_row[2]
            # Fires only if the latest alpha and price equals the one received by the class when signal_event was set
            if self.price == cost or self.close:
                print('Sending order: Iftah Ya Simsim')
                print(f"Side: {side}, Size: {self.size}")
                order = self.binance.create_limit_order(symbol=self.symbol,
                                                        side=side,
                                                        amount=self.size,
                                                        price=cost,
                                                        params={"timeInForce": "GTX", 'postOnly': True})
                self.order_id = order['info']['orderId']
                timestamp = order['lastUpdateTimestamp']
                print(f"Order last updated at {timestamp}")
                #self.price = order['price']
                return True
            else:
                print("Unmatch")
                return False
        except Exception as e:
            print('Exception while placing order: {}'.format(e))
            logging.info('Exception while placing order: {}'.format(e))
            return False


    def trail_order(self):
        """Trails the order at each signal. !!!signal is not the same as alpha!!! Signal keeps track of order trailing"""
        side = 1 if self.alpha[0] == 'buy' else -1
        # Placing the first order
        # Each time there's a new alpha the signal is set, going back to 0 once alpha changes.
        if self.signal == 0:
            self.price = self.alpha[1]
            if self.place_order(self.alpha[0]):
                self.signal = side

    def trade_signal(self):
        """Used for trading once a position has been taken.
        Places order at first alpha and checks at subsequent identical alphas"""
        if self.alpha[0] == 'buy':
            self.trail_order()
            if self.signal == -1:
                self.signal = 0
                self.trail_order()

        elif self.alpha[0] == 'sell':
            self.trail_order()
            if self.signal == 1:
                self.signal = 0
                self.trail_order()


    def run(self):
        """Awaits a signal_event to fire"""
        global alpha, signal_event
        with signal_event:
            while signal_event.wait():  # Wait for a new signal
                # Placing order
                with alpha_lock:
                    self.alpha = alpha
                self.trade_signal()


class DataHandler(Thread):
    """Handles new data from websocket. Runs continuously"""
    THRESHOLD = 0.1

    def open_new_file(self):
        # Calculate model parameters
        try:
            data_date = datetime.utcnow().date() - timedelta(days=1)
            self.params = linear_model(pd.read_csv(f"{self.symbol.lower()}_{data_date}.csv").drop('Time', axis=1), l=1, d=2)
        except Exception as e:
            logging.error(f"Cannot open the file due to: {str(e)}")
            raise e  # re-throw the exception to be caught in the calling function

    def __init__(self, symbol, price_row, price_lock):
        Thread.__init__(self)

        print("New Day")
        self.symbol = symbol
        ## Multiprocessing variables
        self.price_row = price_row
        self.price_lock = price_lock

        ## Data Collection variables
        self.latest_book_ticker = None
        self.latest_agg_trade = None
        self.next_timestamp = None
        self.volume = Decimal('0')
        self.ws = None
        self.file = None
        self.writer = None
        self.row = None
        # Keeping track of new date
        self.date = datetime.utcnow().date()

        ## Signal Generation variables
        self.atp = None
        self.voi = None
        self.oir = None
        self.voi1 = None
        self.oir1 = None
        self.mpb = None
        self.spread = 0.1
        self.params = None

        # Preallocate buffer with NaNs and set current index to 0
        self.buffer = np.full((2, 6), np.NaN)
        self.open_new_file()

    def calculate_next_timestamp(self, timestamp):
        return (timestamp // 100 + 1) * 100

    def cal_metrics(self, row):
        """Calculate the various metrics required for model predictions (mpc)"""
        self.buffer[1] = row
        self.spread = self.buffer[1][2] - self.buffer[1][0]
        diff = np.subtract(self.buffer[1], self.buffer[0])
        dBid = diff[1] if diff[0] == 0 else (self.buffer[1][1] if diff[0] > 0 else 0)
        dAsk = diff[3] if diff[2] == 0 else (self.buffer[1][3] if diff[2] < 0 else 0)
        if self.voi is not None:
            self.voi1 = self.voi
        self.voi = dBid - dAsk
        if self.oir is not None:
            self.oir1 = self.oir
        self.oir = (self.buffer[1][1] - self.buffer[1][3]) / (self.buffer[1][1] + self.buffer[1][3])
        if diff[5] != 0:
            if diff[4] != 0:
                tp = self.buffer[1][4] + diff[5] / diff[4]
            else:
                tp = self.buffer[1][4]
        else:
            tp = self.atp
        self.atp = tp
        m0 = (self.buffer[0][0] + self.buffer[0][2]) / 2
        m1 = (self.buffer[1][0] + self.buffer[1][2]) / 2
        self.mpb = tp - (m0 + m1) / 2

    async def generate_signal(self, row):
        """Calculates the signal based on metrics and sets signal event"""
        global alpha, signal_event
        if np.isnan(self.buffer[0])[0]:
            self.buffer[0] = row
            self.atp = (self.buffer[0][0] + self.buffer[0][2]) / 2
        elif np.isnan(self.buffer[1])[0]:
            self.cal_metrics(row)
        else:
            self.buffer[0] = self.buffer[1]
            self.cal_metrics(row)
            pred_row = np.array([self.mpb, self.oir, self.voi, self.oir1, self.voi1])
            mpc = np.dot(self.params[1:], pred_row / self.spread) + self.params[0]
            # Signal event is set if alpha is buy or sell else event is cleared
            if mpc > self.THRESHOLD:
                print([self.next_timestamp, list(self.row)])
                print("Signal +")
                with alpha_lock:
                    alpha = ['buy', row[0]]
                with signal_event:
                    signal_event.notify()
            elif mpc < (-1 * self.THRESHOLD):
                print([self.next_timestamp, list(self.row)])
                print("Signal -")
                with alpha_lock:
                    alpha = ['sell', row[2]]
                with signal_event:
                    signal_event.notify()

    async def handle_new_message(self, current_event_time):
        """Puts the latest data into queue"""
        if current_event_time < self.next_timestamp and current_event_time >= self.next_timestamp - 100:
            self.row = np.array([self.latest_book_ticker['b'], self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                        self.latest_book_ticker['A'], self.latest_agg_trade['p'], self.volume], dtype='float')
            # Puts new data into the queue at each update
            with self.price_lock:
                self.price_row.put_nowait(self.row)
        elif current_event_time >= self.next_timestamp:
            await self.generate_signal(self.row)
            self.row = np.array([self.latest_book_ticker['b'], self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                        self.latest_book_ticker['A'], self.latest_agg_trade['p'], self.volume], dtype='float')
            # Puts new data into the queue
            with self.price_lock:
                self.price_row.put_nowait(self.row)
            self.next_timestamp += 100

    async def on_message(self, message):
        """Loads data. Awaits handle_new_message"""
        data = orjson.loads(message)
        stream = data['stream']
        event_data = data['data']
        current_event_time = event_data['E']

        if datetime.utcnow().date() != self.date:
            self.__init__(self.symbol, self.price_row, self.price_lock)

        if stream.endswith('@bookTicker'):
            self.latest_book_ticker = event_data
        else:
            self.latest_agg_trade = event_data
            self.volume += Decimal(self.latest_agg_trade['q'])

        if self.latest_book_ticker and self.latest_agg_trade:
            if self.next_timestamp is None:
                self.next_timestamp = self.calculate_next_timestamp(current_event_time)
            await self.handle_new_message(current_event_time)

    async def open_websocket(self):
        """Opens websocket. Awaits response"""
        symbol = self.symbol.lower()
        url = f"wss://fstream.binance.com/stream?streams={symbol}@bookTicker/{symbol}@aggTrade"
        while True:  # Keep trying to connect
            try:
                async with websockets.connect(url) as websocket:
                    while True:
                        response = await websocket.recv()
                        await self.on_message(response)
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
    symbol = 'BTCUSDT'
    # Last in First out queue to get latest price
    price_row = LifoQueue()
    # Lock used to restrict access to shared variables (one thread at a time)
    price_lock = Lock()
    # Although threads are supposed to be functions we can use classes by overriding the Thread class of the Threading package
    # Refer: https://superfastpython.com/extend-thread-class/
    # Same is true for processes
    # Initialize each class
    task1 = DataHandler(symbol, price_row, price_lock)
    task2 = OrderPlacement(symbol, price_row, price_lock)
    task3 = CheckOrder_Panic(symbol)

    # Start each one as a seperate thread
    task1.start()
    task2.start()
    task3.start()

    # Join the threads
    task1.join()
    task2.join()
    task3.join()

if __name__ == '__main__':
    main()