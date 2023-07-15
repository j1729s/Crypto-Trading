import csv
import os
import websocket
import orjson
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from functools import lru_cache
from time import sleep
from Secrets import API_KEY, API_SECRET
from Linear_Data import linear_model
from binance.enums import * # we need these to place an order with binance
from binance.client import Client

getcontext().prec = 11
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect


class DataHandler:
    threshold = 0.1
    BUFFER_SIZE = 2
    def __init__(self):
        self.latest_book_ticker = None
        self.latest_agg_trade = None
        self.next_timestamp = None
        self.volume = Decimal('0')
        self.date = datetime.utcnow().date()
        self.ws = None
        self.file = None
        self.writer = None
        self.params = None

        self.position = 0
        self.atp = None
        self.voi = None
        self.oir = None
        self.voi1 = None
        self.oir1 = None
        self.mpb = None
        self.spread = 0.1
        self.t_volume = 0
        self.cost = 0
        self.client = Client(API_KEY, API_SECRET, tld='com')

        # Preallocate buffer with NaNs and set current index to 0
        self.buffer = np.full((self.BUFFER_SIZE, 6), np.NaN)

    @lru_cache(maxsize=None)
    def calculate_next_timestamp(self, timestamp):
        return ((timestamp + 50) // 100 + 1) * 100

    def handle_new_date(self, new_date):
        self.date = new_date
        self.volume = Decimal('0')  # reset volume
        if self.file:
            price = float(self.latest_agg_trade['p'])
            self.close_trade(-1 * price) if self.position == 1 else self.close_trade(
                -1 * price)
            self.file.close()  # Close the old file
        try:
            self.open_new_file()  # Open a new file
            if self.ws:
                self.ws.close()  # Close the existing WebSocket connection
            self.open_websocket()  # Start a new WebSocket connection
        except Exception as e:
            logging.error(f"Error opening new file: {e}")
            if self.ws:
                self.ws.close()  # Close the websocket if there is an error opening the new file

    def handle_new_message(self, current_event_time):
        if current_event_time >= self.next_timestamp:
            if self.latest_book_ticker and self.latest_agg_trade:
                row = np.array(
                    [self.latest_book_ticker['b'], self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                     self.latest_book_ticker['A'], self.latest_agg_trade['p'], self.volume], dtype='float')
                if np.isnan(self.buffer[0])[0]:
                    self.buffer[0] = row
                    self.atp = (self.buffer[0][0] + self.buffer[0][2]) / 2
                elif np.isnan(self.buffer[1])[0]:
                    self.cal_metrics(row)
                else:
                    self.buffer[0] = self.buffer[1]
                    self.cal_metrics(row)
                    pred_row = np.array([self.mpb, self.oir, self.voi, self.oir1, self.voi1])
                    mpc_pred = np.dot(self.params[1:], pred_row/self.spread) + self.params[0]
                    price = float(self.latest_agg_trade['p'])
                    if self.position == 0:
                        if mpc_pred > self.threshold:
                            self.place_order(row[0], SIDE_BUY)
                            self.open_trade(-1 * row[0])
                        elif mpc_pred < (-1*self.threshold):
                            self.place_order(row[2], SIDE_SELL)
                            self.open_trade(row[2])
                    else:
                        if mpc_pred > self.threshold and self.position == -1:
                            self.place_order(row[0], SIDE_BUY)
                            self.trade(-1 * row[0])
                        elif mpc_pred < (-1*self.threshold) and self.position == 1:
                            self.place_order(row[2], SIDE_SELL)
                            self.trade(row[2])

            self.next_timestamp += 100

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

    def place_order(self, price, side, order_quantity=1, order_type=ORDER_TYPE_LIMIT):
        try:
            print(f'Sending order at {self.next_timestamp}')
            self.client.create_test_order(symbol='BTUSDT', side=side, type=order_type, quantity=order_quantity,
                                                  timeInForce=TIME_IN_FORCE_FOK, price=price)
        except Exception as e:
            print('Exception while placing order: {}'.format(e))
            logging.info('Exception while placing order: {}'.format(e))

    def open_trade(self, price):
        # Opens a position at 'day' start
        self.position = 1 if price < 0 else -1
        self.t_volume += 1
        self.cost = price
        self.write_row()

    def trade(self, price):
        # Trades by longing or shorting two contracts at once
        self.position = 1 if price < 0 else -1
        self.t_volume += 2
        self.cost = 2 * price
        self.write_row()

    def close_trade(self, price):
        # Closes the position at 'day' end
        self.position = 0
        self.t_volume += 1
        self.cost = price
        self.write_row()

    def on_message(self, ws, message):
        data = orjson.loads(message)
        stream = data['stream']
        event_data = data['data']
        current_event_time = event_data['E']

        if stream.endswith('@bookTicker'):
            self.latest_book_ticker = event_data
        else:
            self.latest_agg_trade = event_data
            self.volume += Decimal(self.latest_agg_trade['q'])

        new_date = datetime.utcnow().date()
        if new_date != self.date:
            self.handle_new_date(new_date)

        elif self.next_timestamp is None:
            self.next_timestamp = self.calculate_next_timestamp(current_event_time)

        self.handle_new_message(current_event_time)

    def on_error(self, ws, error):
        logging.error(f"WebSocket encountered an error: {error}")

    def on_close(self, ws, close_status_code=None, close_msg=None):
        logging.info(
            "WebSocket connection closed. Close code: {}. Close message: {}.".format(close_status_code, close_msg))

    def write_row(self):
        row = [self.next_timestamp, self.cost, self.t_volume]
        self.writer.writerow(row)
        print(f"Trade at {row}")
        self.file.flush()  # Force writing of file to disk
        os.fsync(self.file.fileno())  # Make sure it's written to the disk

    def on_open(self, ws):
        logging.info("WebSocket connection established.")

    def open_new_file(self):
        # Reset at day end
        self.__init__()
        # Open the CSV file
        try:
            self.file = open(f'trades1_{self.date}.csv', 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['Time', 'Price', 'Total Volume'])
            data_date = datetime.utcnow().date() - timedelta(days=1)
            self.params = linear_model(pd.read_csv(f"btcusdt_{data_date}.csv").drop('Time', axis=1), l=1, d=2)
        except Exception as e:
            logging.error(f"Cannot open the file due to: {str(e)}")
            raise e  # re-throw the exception to be caught in the calling function

    def open_websocket(self):
        symbol = "btcusdt"
        websocket.enableTrace(False)
        while True:  # Keep trying to connect
            try:
                self.ws = websocket.WebSocketApp(
                    f"wss://fstream.binance.com/stream?streams={symbol}@bookTicker/{symbol}@aggTrade",
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close)
                self.ws.on_open = self.on_open
                self.ws.run_forever(ping_timeout=10)
            except Exception as e:
                logging.error(f"WebSocket encountered an error: {e}")
                logging.info(f"Reconnecting in {RECONNECT_DELAY} seconds...")
                sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    data_handler = DataHandler()
    try:
        data_handler.open_new_file()
        data_handler.open_websocket()
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