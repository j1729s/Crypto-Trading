import csv
import os
import websocket
import orjson
import logging
import numpy as np
from datetime import datetime
from decimal import Decimal, getcontext
from functools import lru_cache
from time import sleep


getcontext().prec = 11
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect


class DataHandler:
    BUFFER_SIZE = 50

    def __init__(self, symbol):
        self.symbol = symbol
        self.latest_book_ticker = None
        self.latest_agg_trade = None
        self.next_timestamp = None
        self.volume = Decimal('0')
        self.date = datetime.utcnow().date()
        self.ws = None
        self.file = None
        self.writer = None
        self.row = None

        # Preallocate buffer with NaNs and set current index to 0
        self.buffer = np.full((self.BUFFER_SIZE, 7), np.nan)
        self.current_index = 0

    @lru_cache(maxsize=None)
    def calculate_next_timestamp(self, timestamp):
        return ((timestamp + 50) // 100 + 1) * 100

    def handle_new_date(self, new_date):
        self.date = new_date
        self.volume = Decimal('0')  # reset volume
        if self.file:
            self.write_buffer()  # Write the remaining data in the buffer to the file
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
        if current_event_time < self.next_timestamp and current_event_time >= self.next_timestamp - 100:
            self.row = [self.next_timestamp, self.latest_book_ticker['b'],
                                               self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                                               self.latest_book_ticker['A'], self.latest_agg_trade['p'],
                                               float(self.volume)]


        elif current_event_time >= self.next_timestamp:
            if self.current_index >= self.BUFFER_SIZE:
                self.writer.writerows(self.buffer)
                # Reset the buffer to NaNs and the current_index to 0
                self.buffer[:] = np.nan
                self.current_index = 0

            self.buffer[self.current_index] = self.row
            self.current_index += 1
            self.next_timestamp += 100
            self.row = [self.next_timestamp, self.latest_book_ticker['b'],
                                               self.latest_book_ticker['B'], self.latest_book_ticker['a'],
                                               self.latest_book_ticker['A'], self.latest_agg_trade['p'],
                                               float(self.volume)]

    def on_message(self, ws, message):
        data = orjson.loads(message)
        stream = data['stream']
        event_data = data['data']
        print(event_data)
        current_event_time = event_data['T']

        new_date = datetime.utcnow().date()
        if new_date != self.date:
            self.handle_new_date(new_date)

        if stream.endswith('@bookTicker'):
            self.latest_book_ticker = event_data
        else:
            self.latest_agg_trade = event_data
            self.volume += Decimal(self.latest_agg_trade['q'])

        if self.latest_book_ticker and self.latest_agg_trade:
            if self.next_timestamp is None:
                self.next_timestamp = self.calculate_next_timestamp(current_event_time)
            self.handle_new_message(current_event_time)

    def on_error(self, ws, error):
        self.write_buffer()
        logging.error(f"WebSocket encountered an error: {error}")

    def on_close(self, ws, close_status_code=None, close_msg=None):
        self.write_buffer()
        logging.info("WebSocket connection closed. Close code: {}. Close message: {}.".format(close_status_code, close_msg))

    def write_buffer(self):
        if self.current_index > 0:
            self.writer.writerows(self.buffer[:self.current_index])
            self.file.flush()  # Force writing of file to disk
            os.fsync(self.file.fileno())  # Make sure it's written to the disk

    def on_open(ws, self):
        logging.info("WebSocket connection established.")

    def open_new_file(self):
        self.__init__(self.symbol)
        # Open the CSV file
        try:
            self.file = open(f'{self.symbol}_{self.date}.csv', 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['Time', 'BestBid', 'BidVol', 'BestAsk', 'AskVol', 'Price', 'Volume'])
        except Exception as e:
            logging.error(f"Cannot open the file due to: {str(e)}")
            raise e  # re-throw the exception to be caught in the calling function

    def open_websocket(self):
        global RECONNECT_DELAY
        symbol = self.symbol
        websocket.enableTrace(False)
        while True:  # Keep trying to connect
            try:
                self.ws = websocket.WebSocketApp(
                    f"wss://stream.binance.com:9443/stream?streams={symbol}@bookTicker/{symbol}@aggTrade",
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
    symbol = 'btcusdt'
    data_handler = DataHandler(symbol)
    try:
        data_handler.open_new_file()
        data_handler.open_websocket()
    except KeyboardInterrupt:
        logging.info("\nInterrupted.")
    finally:
        logging.info("Closing file and connection...")
        if data_handler.file:
            data_handler.write_buffer()  # Write the remaining data in the buffer to the file
            data_handler.file.flush()  # Force writing of file to disk
            os.fsync(data_handler.file.fileno())  # Make sure it's written to the disk
            data_handler.file.close()
        if data_handler.ws:
            data_handler.ws.close()  # check if the websocket is open before calling close