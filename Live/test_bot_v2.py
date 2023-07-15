import os
import csv
import ccxt
import asyncio
import websockets
import orjson
import logging
import time as t
import numpy as np
import pandas as pd
from time import sleep
from functools import lru_cache
from Linear_Data import linear_model
from decimal import Decimal, getcontext
from Secrets import API_KEY, API_SECRET
from datetime import datetime, timedelta, time
from pathos.parallel import ParallelPool
from pathos.threading import ThreadPool

getcontext().prec = 11
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect

class DataHandler:
    threshold = 0.1
    BUFFER_SIZE = 2
    CONTRACT_SIZE = 0.002
    def __init__(self, symbol):
        self.symbol = symbol
        self.latest_book_ticker = None
        self.latest_agg_trade = None
        self.next_timestamp = None
        self.volume = Decimal('0')
        self.date = datetime.utcnow().date()
        self.clock = round(t.time()) + self.calculate_runtime(self.date)
        self.ws = None
        self.file = None
        self.writer = None
        self.params = None
        self.row = None

        self.atp = None
        self.voi = None
        self.oir = None
        self.voi1 = None
        self.oir1 = None
        self.mpb = None
        self.spread = 0.1
        self.alpha = None

        self.position_portfolio = 0
        self.position_trade = 0
        self.balance = self.CONTRACT_SIZE
        self.open = False
        self.side = None
        self.signal = 0
        self.size = self.CONTRACT_SIZE
        self.binance = ccxt.binanceusdm({'apiKey': API_KEY,
                                         'secret': API_SECRET})

        self.binance.load_markets()  # load markets to get the market id from a unified symbol
        self.binance.setLeverage(2, symbol=self.symbol)

        # Preallocate buffer with NaNs and set current index to 0
        self.buffer = np.full((self.BUFFER_SIZE, 6), np.NaN)

    def calculate_runtime(self, date_now):
        midnight = datetime.combine(date_now + timedelta(days=1), time(0, 0, 0))
        return (midnight - datetime.utcnow()).total_seconds() - 300

    @lru_cache(maxsize=None)
    def calculate_next_timestamp(self, timestamp):
        return (timestamp // 100 + 1) * 100

    def handle_new_date(self):
        if self.file:
            self.file.close()  # Close the old file
        if self.ws:
            self.ws.close()  # Close the existing WebSocket connection
        while datetime.utcnow().date() == self.date:
            sleep(RECONNECT_DELAY)
        try:
            self.open_new_file()  # Open a new file
            self.open_websocket()  # Start a new WebSocket connection
        except Exception as e:
            logging.error(f"Error opening new file: {e}")



    def cal_metrics(self, row):
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
                tp = self.buffer[1][4] + diff[5]/diff[4]
            else:
                tp = self.buffer[1][4]
        else:
            tp = self.atp
        self.atp = tp
        m0 = (self.buffer[0][0] + self.buffer[0][2]) / 2
        m1 = (self.buffer[1][0] + self.buffer[1][2]) / 2
        self.mpb = tp - (m0 + m1) / 2

    def write_row(self, price, side):
        row = [self.next_timestamp, price, self.size, side]
        self.writer.writerow(row)
        print(f"Trade at {row}")
        print(self.position_portfolio, self.position_trade, self.signal, self.side, self.balance, self.open)
        self.file.flush()  # Force writing of file to disk
        os.fsync(self.file.fileno())  # Make sure it's written to the disk

    def check_order(self):
        try:
            check = self.binance.fetchOrder(self.order_id, symbol = self.symbol, params = {})
            if check['status'] == 'closed':
                print('Filled')
                quant = float(check['filled'])
                self.position_trade += self.side * quant
                self.size = 0
                self.open = True
            elif check['status'] == 'open':
                cancel = self.binance.cancelOrder(self.order_id, symbol = self.symbol, params = {})
                quant = float(cancel['filled'])
                self.position_trade += self.side * quant
                self.size -= quant
        except Exception as e:
            print('Exception while placing order: {}'.format(e))
            logging.info('Exception while placing order: {}'.format(e))

    def place_order(self, side):
        try:
            last_price = self.queue_price.get_nowait()
            last_signal = self.queue_signal.get_nowait()
            cost = last_price[0] if side == 'buy' else last_price[1]
            if last_signal[0] == self.alpha and last_signal[1] == cost:
                print(f'Sending order at {self.next_timestamp}')
                print(cost, side, self.size)
                order = self.binance.create_limit_order(symbol=self.symbol,
                                                        side=side,
                                                        amount=self.size,
                                                        price=cost,
                                                        params={"timeInForce": "GTX", 'postOnly': True})
                self.order_id = order['info']['orderId']
                print(order['price'])
                return True
        except Exception as e:
            print('Exception while placing order: {}'.format(e))
            logging.info('Exception while placing order: {}'.format(e))
            return False

    def generate_signal(self, row):
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
            if mpc > self.threshold:
                self.alpha = 'buy'
                self.queue_signal.put_nowait((self.alpha,row[0]))
            elif mpc < (-1 * self.threshold):
                self.alpha = 'sell'
                self.queue_signal.put_nowait((self.alpha,row[2]))
            else:
                self.alpha = None

    def update_trade_metrics(self, type):
        print(self.position_trade)
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
        price = self.row[0] if self.alpha == 'buy' else self.row[2]
        side = 1 if self.alpha == 'buy' else -1
        # Placing the first order
        if self.signal == 0:
            self.size = size
            if self.place_order(self.alpha):
                self.side = side
                self.signal = side
                self.write_row(price, self.alpha.upper())
        # Checking order at each (matching) signal if unfilled we cancel and repost else we update metrics
        elif self.signal == side:
            self.check_order()
            if self.size > 0:
                self.place_order(self.alpha)
            else:
                self.update_trade_metrics(type)
            self.write_row(price, self.alpha.upper())

    def open_trade(self, row):
        # Assume a series of alphas like 1,1,1,1,...,-1,-1,-1,...,1,1,... or the mirror alphas
        if self.alpha == 'buy':
            self.trail_order(self.size, 'open')
            # Cancelling and updating in case the alpha changes. Here we only want to cancel.
            # In case its partially filled, we have opened else we wait for the next signal
            if self.signal == -1 and self.size > 0:
                self.check_order()
                self.update_trade_metrics('open')
                if self.position_portfolio != 0:
                    self.open = True
                self.write_row(row[2], 'SELL')

        elif self.alpha == 'sell':
            self.trail_order(self.size, 'open')
            # See the comment above
            if self.signal == 1 and self.size > 0:
                self.check_order()
                self.update_trade_metrics('open')
                if self.position_portfolio != 0:
                    self.open = True
                self.write_row(row[0], 'BUY')

    def close_trade(self):
        # In case of an open order
        if self.signal != 0:
            self.check_order()
            self.update_trade_metrics('trade')
        if self.position_portfolio != 0:
            self.size = abs(self.position_portfolio)
            if self.position_portfolio > 0:
                self.alpha = 'sell'
            else:
                self.alpha = 'buy'
            self.trail_order(self.size, 'trade')

    def trade_signal(self, row):
        # Here in case of an unfilled order we want to cancel and place another order at the same iteration
        if self.alpha == 'buy' and self.position_portfolio <= 0:
            if self.signal == -1 and self.size > 0:
                self.check_order()
                self.update_trade_metrics('')
                self.write_row(row[2], 'SELL')
            size = min((2 * self.CONTRACT_SIZE), self.balance)
            self.trail_order(size, 'trade')

        elif self.alpha == 'sell' and self.position_portfolio >= 0:
            if self.signal == 1 and self.size > 0:
                self.check_order()
                self.update_trade_metrics('')
                self.write_row(row[0], 'BUY')
            size = 2 * self.CONTRACT_SIZE
            self.trail_order(size, 'trade')

    def process_signal(self):
        if self.alpha is not None:
            # Placing opening order
            if not self.open and self.position_portfolio == 0:
                self.open_trade(self.row)
            # Trading
            if self.open:
                self.trade_signal(self.row)

    async def handle_new_message(self, current_event_time):
        if current_event_time < self.next_timestamp and current_event_time >= self.next_timestamp - 100:
            self.row = np.array(
                [self.latest_book_ticker['b'], self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                 self.latest_book_ticker['A'], self.latest_agg_trade['p'], self.volume], dtype='float')
            self.queue_price.put_nowait((self.row[0], self.row[2]))

        elif current_event_time >= self.next_timestamp:
            # Closing and sleeping for a while a day end
            if t.time() >= self.clock:
                if self.position_portfolio != 0:
                    self.close_trade()
                else:
                    self.handle_new_date()

            print([self.next_timestamp, list(self.row)])
            self.generate_signal(self.row)
            self.process_signal()
            self.row = np.array(
                [self.latest_book_ticker['b'], self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                 self.latest_book_ticker['A'], self.latest_agg_trade['p'], self.volume], dtype='float')
            self.queue_price.put_nowait((self.row[0], self.row[2]))
            self.next_timestamp += 100


    async def on_message(self, message):
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

    def open_new_file(self):
        # Reset at day end
        self.__init__(self.symbol)
        # Open the CSV file
        try:
            self.file = open(f'1test_btc_{self.date}.csv', 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['Time', 'Price', 'Quantity', 'Side'])
            data_date = datetime.utcnow().date() - timedelta(days=11)
            self.params = linear_model(pd.read_csv(f"btcusdt_{data_date}.csv").drop('Time', axis=1), l=1, d=2)
        except Exception as e:
            logging.error(f"Cannot open the file due to: {str(e)}")
            raise e  # re-throw the exception to be caught in the calling function

    async def open_websocket(self):
        symbol = self.symbol.lower()
        url = f"wss://fstream.binance.com/stream?streams={symbol}@bookTicker/{symbol}@aggTrade"
        while True:  # Keep trying to connect
            async with websockets.connect(url) as websocket:
                while True:
                    response = await websocket.recv()
                    await self.on_message(response)

    async def main(self):

        tasks = [
            asyncio.create_task(self.open_websocket())
        ]

        await tasks[0]
if __name__ == "__main__":
    data_handler = DataHandler('BTCUSDT')
    try:
        data_handler.open_new_file()
        asyncio.run(data_handler.main())
    except KeyboardInterrupt:
        logging.info("\nInterrupted.")
    finally:
        logging.info("Closing file and connection...")
        if data_handler.file:
            data_handler.file.flush()  # Force writing of file to disk
            os.fsync(data_handler.file.fileno())  # Make sure it's written to the disk
            data_handler.file.close()
        if data_handler.ws:
            data_handler.ws.close()  # check if the websocket is open before calling close