import websockets
import json
import asyncio
import csv
import os
import logging
import numpy as np
from datetime import datetime
from time import sleep


logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

RECONNECT_DELAY = 5  # Delay (in seconds) before attempting to reconnect


class DataHandler:
    BUFFER_SIZE = 50

    def __init__(self):
        self.latest_book_ticker = None
        self.buy_vol = 0
        self.sell_vol = 0
        self.volume = 0
        self.date = datetime.utcnow().date()
        self.file = None
        self.writer = None
        self.row = None

        # Preallocate buffer with NaNs and set current index to 0
        self.buffer = np.full((self.BUFFER_SIZE, 5), np.nan)
        self.current_index = 0

    def handle_new_date(self, new_date):
        self.date = new_date
        self.volume = 0  # reset volume
        if self.file:
            self.write_buffer()  # Write the remaining data in the buffer to the file
            self.file.close()  # Close the old file
        try:
            self.open_new_file()  # Open a new file
        except Exception as e:
            logging.error(f"Error opening new file: {e}")

    def handle_new_message(self):
        side = 1 if self.latest_book_ticker['side'] == 'buy' else -1
        self.row = [self.latest_book_ticker['ts'], self.latest_book_ticker['px'],
                                               self.latest_book_ticker['sz'], side,
                                               self.latest_book_ticker['count']]
        print(self.row)
        if self.current_index >= self.BUFFER_SIZE:
            self.writer.writerows(self.buffer)
            # Reset the buffer to NaNs and the current_index to 0
            self.buffer[:] = np.nan
            self.current_index = 0

        # print(self.row)
        self.buffer[self.current_index] = self.row
        self.current_index += 1

    def on_message(self, data):
        new_date = datetime.utcnow().date()
        if new_date != self.date:
            self.handle_new_date(new_date)

        self.latest_book_ticker = data['data'][0]

        if self.latest_book_ticker is not None:
            self.handle_new_message()

    def on_close(self, close_status_code=None, close_msg=None):
        self.write_buffer()
        logging.info("WebSocket connection closed. Close code: {}. Close message: {}.".format(close_status_code, close_msg))

    def write_buffer(self):
        if self.current_index > 0:
            self.writer.writerows(self.buffer[:self.current_index])
            self.file.flush()  # Force writing of file to disk
            os.fsync(self.file.fileno())  # Make sure it's written to the disk

    def open_new_file(self):
        self.__init__()
        # Open the CSV file
        try:
            self.file = open(f'okx_btcusdt_trades_{self.date}.csv', 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['Time', 'Price', 'Size', 'Side', 'Count'])
        except Exception as e:
            logging.error(f"Cannot open the file due to: {str(e)}")
            raise e  # re-throw the exception to be caught in the calling function

async def handle_websocket_connection(symbol):
    url = "wss://ws.okx.com:8443/ws/v5/public"
    data_handler = DataHandler()
    data_handler.open_new_file()
    while True:
        try:
            async with websockets.connect(url) as websocket:
                # Subscribe to the book ticker stream for the BTC-USDT trading pair
                book_ticker_request = {
                    "op": "subscribe",
                    "args": [
                        {"channel": "trades",
                         "instId": symbol,
                         },
                    ]
                }
                await websocket.send(json.dumps(book_ticker_request))
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    if 'data' in data.keys():
                        data_handler.on_message(data)
        except Exception as e:
            logging.error(f"WebSocket encountered an error: {e}")
            logging.info(f"Reconnecting in {RECONNECT_DELAY} seconds...")
            sleep(RECONNECT_DELAY)


# Run the event loop
async def main(symbol):
    task = asyncio.create_task(handle_websocket_connection(symbol))
    await task


if __name__ == "__main__":
    symbol = "BTC-USDT-SWAP"
    asyncio.run(main(symbol))