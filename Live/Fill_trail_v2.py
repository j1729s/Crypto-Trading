import ccxt
import asyncio
import websockets
import orjson
import logging
import time as t
import numpy as np
import pandas as pd
from time import sleep
from Linear_Data import linear_model
from decimal import Decimal, getcontext
from Secrets import API_KEY, API_SECRET
from datetime import datetime, timedelta, time

from threading import Lock, Event, Thread, Condition
from queue import LifoQueue

getcontext().prec = 11
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect

# Define Global Variables
alpha = [None, None]
alpha_lock = Lock()
# Event (boolean operator) used to fire process by setting or clearing
data_event = Condition()
signal_event = Condition()
close_event = Condition()
new_day_event = Event()

class GenerateSignal(Thread):
    """Concerned with calculating signal based on latest price data got from the queue.
    Fires when data_event is set"""
    THRESHOLD = 0.1
    BUFFER_SIZE = 2

    def open_new_file(self):
        # Calculate model parameters
        try:
            data_date = datetime.utcnow().date() - timedelta(days=1)
            self.params = linear_model(pd.read_csv(f"{self.symbol}_{data_date}.csv").drop('Time', axis=1), l=1, d=2)
        except Exception as e:
            logging.error(f"Cannot open the file due to: {str(e)}")
            raise e  # re-throw the exception to be caught in the calling function


    def __init__(self, symbol, price_row, price_lock):
        Thread.__init__(self)

        self.symbol = symbol.lower()

        ## Multithreading variables
        self.price_row = price_row
        self.price_lock = price_lock
        #self.new_data_event = data_event
        #self.new_signal_event = signal_event

        ## Signal Generation variables
        self.atp = None
        self.voi = None
        self.oir = None
        self.voi1 = None
        self.oir1 = None
        self.mpb = None
        self.spread = 0.1
        self.params = None

        self.date = datetime.utcnow().date()
        # Preallocate buffer with NaNs and set current index to 0
        self.buffer = np.full((self.BUFFER_SIZE, 6), np.NaN)
        self.open_new_file()

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

    def generate_signal(self, row):
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
                print("signal +")
                with alpha_lock:
                    alpha = ['buy', row[0]]
                with signal_event:
                    signal_event.notify()
            elif mpc < (-1 * self.THRESHOLD):
                print("signal -")
                with alpha_lock:
                    alpha = ['sell', row[2]]
                with signal_event:
                    signal_event.notify()


    def run(self):
        global data_event
        """Awaits new data using data_event variable, gets latest price from lifo queue,
         generates signal and clears data_event"""
        with data_event:
            while data_event.wait():  # Wait for new data
                with self.price_lock:
                    row = self.price_row.get_nowait()
                self.generate_signal(row)
                if datetime.utcnow().date() != self.date:
                    self.__init__(self.price_row, self.price_lock)
                # data_event.clear()  # Reset the data event


class OrderPlacement(Thread):
    """Concerned with order placement. Fires when signal_event is set"""
    CONTRACT_SIZE = 0.001

    def __init__(self, symbol, price_row, price_lock):
        Thread.__init__(self)
        ## Multithreading variables
        self.price_lock = price_lock
        self.price_row = price_row
        #self.new_signal_event = signal_event
        ## Trading Metrics
        self.alpha = None
        self.symbol = symbol
        self.position_portfolio = 0
        self.position_trade = 0
        self.balance = self.CONTRACT_SIZE
        self.open = False
        self.side = None
        self.signal = 0
        self.size = self.CONTRACT_SIZE
        self.order_id = None
        self.close = False
        self.price = 0.0
        self.cancel = False

        ## Exchange info and constants
        self.binance = ccxt.binanceusdm({'apiKey': API_KEY,
                                         'secret': API_SECRET})
        self.binance.load_markets()  # load markets to get the market id from a unified symbol
        self.binance.setLeverage(11, symbol=self.symbol)  # set account cross leverage

    def check_order(self):
        """Checks and cancels open orders"""
        try:
            with self.price_lock:
                last_row = self.price_row.get_nowait()
            cost = last_row[0] if self.alpha[0] == 'buy' else last_row[2]
            last_signal_cost = last_row[0] if self.alpha[0] == 'sell' else last_row[2]
            if self.cancel == True and self.price == last_signal_cost:
                print("ere")
                self.cancel = False
                return False
            elif self.price != cost:
                print("here")
                check = self.binance.fetchOrder(self.order_id, symbol=self.symbol, params={})
                if check['status'] == 'closed':
                    print('Filled')
                    quant = float(check['filled'])
                    self.position_trade += self.side * quant
                    self.size = 0
                    self.open = True
                    self.cancel = False
                    # we want to update in this case
                    return True
                elif check['status'] == 'open':
                    cancel = self.binance.cancelOrder(self.order_id, symbol=self.symbol, params={})
                    print("Cancelled")
                    quant = float(cancel['filled'])
                    self.position_trade += self.side * quant
                    self.size -= quant
                    self.price = cost
                    self.cancel = False
                    # we want to place order or update
                    return True
                else:  # in case pf camcelled order by binance
                    self.price = cost # updating in case the order had already been cancelled
                    return True
            else:  # we don't check unless price changes
                return False
        except Exception as e:
            print('Exception while checking order: {}'.format(e))
            logging.info('Exception while checking order: {}'.format(e))
            return False

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

    def update_trade_metrics(self, type):
        """Keeps track of portfolio metrics like position size and balance etc"""
        # For initial trade (At open)
        if type == 'open':
            self.position_portfolio = self.position_trade
        # While Trading
        else:
            self.position_portfolio += self.position_trade
        self.balance -= self.position_trade
        self.signal = 0
        self.position_trade = 0

    def trail_order(self, size, type):
        """Trails the order at each signal. !!!signal is not the same as alpha!!! Signal keeps track of order trailing"""
        side = 1 if self.alpha[0] == 'buy' else -1
        # Placing the first order
        # Each time there's a new alpha the signal is set, going back to 0 once alpha changes.
        if self.signal == 0:
            self.price = self.alpha[1]
            self.size = size
            if self.place_order(self.alpha[0]):
                self.side = side
                self.signal = side
        # Checking order at each (matching) signal if unfilled we cancel and repost else we update metrics
        elif self.signal == side:
            if self.check_order():
                if self.size > 0:
                    self.place_order(self.alpha[0])
                else:
                    self.update_trade_metrics(type)


    def open_trade(self):
        """Used for taking the opening position each day.
        Places order at first alpha and checks at subsequent identical alphas"""
        # Assume a series of alphas like 1,1,1,1,...,-1,-1,-1,...,1,1,... or the mirror alphas
        if self.alpha[0] == 'buy':
            self.trail_order(self.size, 'open')
            # Cancelling and updating in case the alpha changes. Here we only want to cancel.
            # In case its partially filled, we have opened else we wait for the next signal
            if self.signal == -1 and self.size > 0:
                self.cancel = True
                if self.check_order():
                    self.update_trade_metrics('open')
                    if self.position_portfolio != 0:
                        self.open = True


        elif self.alpha[0] == 'sell':
            self.trail_order(self.size, 'open')
            # See the comment above
            if self.signal == 1 and self.size > 0:
                self.cancel = True
                if self.check_order():
                    self.update_trade_metrics('open')
                    if self.position_portfolio != 0:
                        self.open = True


    def close_trade(self):
        """Used for closing. Not relevant atm"""
        global new_day_event
        # In case of an open order
        if self.signal != 0:
            self.check_order()
            self.update_trade_metrics('trade')
        if self.position_portfolio != 0:
            self.size = abs(self.position_portfolio)
            if self.position_portfolio > 0:
                self.alpha[0] = 'sell'
            else:
                self.alpha[0] = 'buy'
            self.trail_order(self.size, 'trade')
        elif self.position_portfolio == 0:
            print("Closed")
            with new_day_event:
                new_day_event.set()

    def trade_signal(self):
        """Used for trading once a position has been taken.
        Places order at first alpha and checks at subsequent identical alphas"""
        # Here in case of an unfilled order we want to cancel and place another order at the same iteration
        if self.alpha[0] == 'buy':  # cannot seperate the two conditions, might not be a problem becase trade position oes to zero at each update
            if self.signal == -1 and self.size > 0:
                self.cancel = True
                if self.check_order():  # because it will update twice in case of alternating signal like open with buy and then another buy signal arrives
                    self.update_trade_metrics('')
                    self.price = self.alpha[1]
            if self.position_portfolio <= 0:
                size = min((2 * self.CONTRACT_SIZE), self.balance)
                self.trail_order(size, 'trade')

        elif self.alpha[0] == 'sell':
            if self.signal == 1 and self.size > 0:
                self.cancel = True
                if self.check_order():
                    self.update_trade_metrics('')
                    self.price = self.alpha[1]
            if self.position_portfolio >= 0:
                size = min((2 * self.CONTRACT_SIZE), (2 * self.CONTRACT_SIZE - self.balance))
                self.trail_order(size, 'trade')

    def run(self):
        """Awaits a signal_event to fire"""
        global alpha, signal_event, close_event
        with signal_event:
            while signal_event.wait():  # Wait for a new signal
                # Placing opening order
                with alpha_lock:
                    self.alpha = alpha
                # Decides how to trade (open or else)
                if not self.open and self.position_portfolio == 0:
                    self.open_trade()
                # Trading
                if self.open:
                    self.trade_signal()

        #with close_event:
            #while close_event.wait():
                #self.close = True
                #print("Closing")
                #self.close_trade()
                #sleep(RECONNECT_DELAY)



class DataHandler(Thread):
    """Handles new data from websocket. Runs continuously"""
    def calculate_runtime(self, date_now):
        """Calculates time remaining to UTC midnight. Used for closing"""
        midnight = datetime.combine(date_now + timedelta(days=1), time(0, 0, 0))
        return (midnight - datetime.utcnow()).total_seconds() - 300  # 300 seconds i.e., closes 5 mins before day end

    def __init__(self, symbol, price_row, price_lock):
        Thread.__init__(self)
        self.symbol = symbol
        ## Multiprocessing variables
        self.price_row = price_row
        self.price_lock = price_lock
        #self.new_data_event = data_event

        ## Data Collection variables
        self.latest_book_ticker = None
        self.latest_agg_trade = None
        self.next_timestamp = None
        self.volume = Decimal('0')
        self.ws = None
        self.file = None
        self.writer = None
        self.row = None
        # Keeping track of new date and closing at day end
        self.date = datetime.utcnow().date()
        self.clock = round(t.time()) + 30#self.calculate_runtime(self.date)

    def calculate_next_timestamp(self, timestamp):
        return (timestamp // 100 + 1) * 100

    def handle_new_date(self):
        """Not relevant"""
        print("Sleeping")
        #while datetime.utcnow().date() == self.date:
            #sleep(RECONNECT_DELAY)
        main()


    async def handle_new_message(self, current_event_time):
        global data_event, close_event, new_day_event
        """Puts the latest data into queue"""
        if current_event_time < self.next_timestamp and current_event_time >= self.next_timestamp - 100:
            self.row = np.array([self.latest_book_ticker['b'], self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                        self.latest_book_ticker['A'], self.latest_agg_trade['p'], self.volume], dtype='float')
            # Puts new data into the queue at each update
            with self.price_lock:
                self.price_row.put_nowait(self.row)
        elif current_event_time >= self.next_timestamp:
            print([self.next_timestamp, list(self.row)])
            # Closing and sleeping for a while a day end. Not relevant used for closing
            if datetime.utcnow().date() != self.date:
                self.__init__(self.symbol, self.price_row, self.price_lock)
            #if t.time() >= self.clock:
                #print("time")
                #if new_day_event.isSet():
                    #self.handle_new_date()
                #with close_event:
                    #print("close")
                    #close_event.notify()
            # This controls when generate_signal is fired. Currently, it's every 100ms.
            with data_event:
                data_event.notify()

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
    print("New Day")
    # Although threads are supposed to be functions we can use classes by overriding the Thread class of the Threading package
    # Refer: https://superfastpython.com/extend-thread-class/
    # Same is true for processes
    # Initialize each class
    task1 = DataHandler(symbol, price_row, price_lock)
    task2 = GenerateSignal(symbol, price_row, price_lock)
    task3 = OrderPlacement(symbol, price_row, price_lock)

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