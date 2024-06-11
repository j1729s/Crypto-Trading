import json
import decimal
import asyncio
import logging
import websockets
import numpy as np
import statsmodels.api as sm
from Bybit_GLFT_Helper import CircularBuffer, OrderPlacement, GetPosition
from Bybit_Constants import *

decimal.getcontext().prec = 10
PING_ID = 1729

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class DataHandler:
    BUFFER_SIZE = 50

    def __init__(self):
        self.file = None
        self.writer = None
        self.row = None
        self.vol = 0.0
        self.last_array = None
        self.counter = 0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.mid = 0.0
        self.position = None
        self.bid_hs = 0.0
        self.bid_skew = 0.0
        self.ask_hs = 0.0
        self.ask_skew = 0.0
        self._mid_lock = asyncio.Lock()
        self._pos_lock = asyncio.Lock()
        self.buffer = CircularBuffer(NUM_ROWS, NUM_COLS)
        self.Order_System = OrderPlacement()
        self.Position_System = GetPosition()

    @staticmethod
    def cal_vol_p(arr):
        yerr = (arr - arr[0])[1:]
        return np.nanmean(np.abs(yerr - np.nanmean(yerr))) * np.sqrt(3.14 / 2)

    @staticmethod
    def cal_mad(arr):
        yerr = arr - np.roll(arr, 1)
        return np.nanmean(np.abs(yerr - np.nanmean(yerr))) * np.sqrt(3.14 / 2)

    @staticmethod
    def compute_coeff(xi, gamma, delta, A, k):
        inv_k = np.divide(1, k)
        c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
        c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
        return c1, c2

    @staticmethod
    def get_reg_coeff(data, ticks):
        # Add a constant term for the intercept
        X_with_intercept = sm.add_constant(ticks)
        # Fit the OLS model
        print(data)
        model = sm.OLS(np.log(data), X_with_intercept)
        print(np.log(data))
        result = model.fit()
        print("Params are: ", result.params)
        A = np.exp(result.params[0])
        k = -1 * result.params[1]
        return A, k

    def cal_spread_skew(self, side_dict, ticks, vol_r):
        A, k = self.get_reg_coeff(list(side_dict.values()), ticks)
        c1, c2 = self.compute_coeff(GAMMA, GAMMA, DELTA, A, k)
        half_spread = 1 * c1 + 1 / 2 * c2 * vol_r
        skew = c2 * vol_r
        return half_spread, skew

    async def handle_level1(self, data):
        async with self._mid_lock:
            if len(data['b']) > 0:
                self.best_bid = float(data['b'][0][0])
            if len(data['a']) > 0:
                self.best_ask = float(data['a'][0][0])
            self.mid = round((self.best_bid + self.best_ask) / 2, 2)

    async def handle_trade_stream(self, data):
        dummy_list = [[dat['T'], 1, dat['p'], dat['v']] if dat['S'] == 'Buy' else [dat['T'], -1, dat['p'], dat['v']] for dat in data]
        self.buffer.append(np.array(dummy_list))
        shape = self.buffer.get().shape[0]
        if shape == NUM_ROWS:
            if self.counter % 100 == 0:
                position_dict = await self.Position_System.get_positions()
                position = position_dict['result']['list'][0]['size']
                side = position_dict['result']['list'][0]['side']
                async with self._pos_lock:
                    self.position = float(position) * -1.0 if side == 'Sell' else float(position)
                await self.Order_System.cancel_orders()
                await self.cal_intensity()
            self.counter += 1
        elif shape % 100 == 0:
            print(shape)


    @staticmethod
    def process_chunks(reshaped_data, price_thresh, num_chunks, tick, side='bid'):
        # Apply the conditions
        if side == 'bid':
            condition1 = reshaped_data[:, :, 1] == -1
            price_thresh_broadcasted = np.broadcast_to(price_thresh[:, np.newaxis], (num_chunks, CHUNK_SIZE))
            condition2 = reshaped_data[:, :, 2] <= np.round(price_thresh_broadcasted - tick, 1)
        else:
            condition1 = reshaped_data[:, :, 1] == 1
            price_thresh_broadcasted = np.broadcast_to(price_thresh[:, np.newaxis], (num_chunks, CHUNK_SIZE))
            condition2 = reshaped_data[:, :, 2] >= np.round(price_thresh_broadcasted + tick, 1)

        # Combine the conditions
        combined_condition = condition1 & condition2

        # Find the first occurrence index for each chunk where the condition is met
        first_occurrence_indices = np.argmax(combined_condition, axis=1)

        # Check if the condition is never met in each chunk
        no_condition_met = ~combined_condition.any(axis=1)

        # Replace indices where the condition is never met with the index of the last row
        first_occurrence_indices[no_condition_met] = reshaped_data.shape[1] - 1

        # Gather the required rows using advanced indexing
        rows = np.arange(reshaped_data.shape[0])
        result = reshaped_data[rows, first_occurrence_indices]
        return result[:, 0]

    @staticmethod
    async def retry_orders(mid, best_bid, best_ask, position, bid_hs, bid_skew, ask_hs, ask_skew):
        print("Retrying")
        print(mid, best_bid, best_ask)
        bid = np.round((mid - bid_hs - bid_skew * int(position / (MIN_ORD_QUANT * DELTA))), 1)
        bid = best_bid - FIXED_SPREAD if bid > best_bid - FIXED_SPREAD else bid
        ask = np.round((mid + ask_hs - ask_skew * int(position / (MIN_ORD_QUANT * DELTA))), 1)
        ask = best_ask + FIXED_SPREAD if ask < best_ask + FIXED_SPREAD else ask
        return [bid, ask]

    async def cal_intensity(self):
        data = self.buffer.get()
        data = data[data[:, 0].argsort()]
        # Reshape the data into chunks
        num_chunks = data.shape[0] // CHUNK_SIZE
        reshaped_data = data[:num_chunks * CHUNK_SIZE].reshape(num_chunks, CHUNK_SIZE, -1)

        # Extract time_thresh (first element of the first row in each chunk)
        time_thresh = reshaped_data[:, 0, 0]
        price_thresh = reshaped_data[:, 0, 2]
        avg_move_up = np.mean([np.max(chunk[:, 2]) for i, chunk in enumerate(reshaped_data)] - price_thresh)
        avg_move_dn = np.mean(price_thresh - [np.min(chunk[:, 2]) for i, chunk in enumerate(reshaped_data)])
        print(f"Move up is {avg_move_up}, move down is {avg_move_dn}")
        print("Counter is: ", self.counter)
        ticks = np.arange(5, 30, 2.5) / 10
        bid_mle = {}
        ask_mle = {}
        for tick in ticks:
            bid_mle[tick] = 1 / np.mean((self.process_chunks(reshaped_data, price_thresh, num_chunks, tick) - time_thresh) / 1_000)
            ask_mle[tick] = 1 / np.mean((self.process_chunks(reshaped_data, price_thresh, num_chunks, tick, 'ask') - time_thresh) / 1_000)
        print("Bid MLE: ", bid_mle)
        print("Ask MLE: ", ask_mle)
        self.vol = self.cal_mad(reshaped_data[-1, :, 2])
        print("Volatility is: ", self.vol)
        print("Bid Side: \n")
        self.bid_hs, self.bid_skew = self.cal_spread_skew(bid_mle, ticks, self.vol)
        print('half_spread={}, skew={}'.format(self.bid_hs, self.bid_skew))
        async with self._pos_lock:
            position = self.position
        async with self._mid_lock:
            bid = np.round((self.mid - self.bid_hs - self.bid_skew * int(position / (MIN_ORD_QUANT * DELTA))), 1)
            bid = self.best_bid - FIXED_SPREAD if bid > self.best_bid - FIXED_SPREAD else bid
        print("Ask Side: \n")
        self.ask_hs, self.ask_skew = self.cal_spread_skew(ask_mle, ticks, self.vol)
        print('half_spread={}, skew={}'.format(self.ask_hs, self.ask_skew))
        async with self._mid_lock:
            ask = np.round((self.mid + self.ask_hs - self.ask_skew * int(self.position / (MIN_ORD_QUANT * DELTA))), 1)
            ask = self.best_ask + FIXED_SPREAD if ask < self.best_ask + FIXED_SPREAD else ask
        print("Position is", self.position)
        print(bid, ask)
        async with self._mid_lock:
            print(self.best_bid, self.best_ask)
        isRetry = await self.Order_System.place_order([bid, ask])
        """
        async with self._pos_lock:
            position_new = self.position
        if isRetry and position_new == position:
            while isRetry:
                async with self._mid_lock:
                    mid = self.mid
                    bb = self.best_bid
                    ba = self.best_ask
                array = await self.retry_orders(mid, bb, ba, position, self.bid_hs, self.bid_skew, self.ask_hs, self.ask_skew)
                await self.Order_System.cancel_orders()
                isRetry = await self.Order_System.place_order(array)
                await asyncio.sleep(0.5)
        """

    async def on_message(self, data):
        if data['topic'] == 'publicTrade.BTCUSDT':
            await self.handle_trade_stream(data['data'])
        else:
            await self.handle_level1(data['data'])


async def send_ping(market='spot'):
    url = f"wss://stream.bybit.com/v5/public/{market}"
    ping_request = {
        "req_id": f"{PING_ID}",
        "op": "ping",
    }
    while True:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps(ping_request))
            await asyncio.sleep(13)


# Run the event loop
async def websocket_client(symbol, market='spot'):
    data_handler = DataHandler()
    url = f"wss://stream.bybit.com/v5/public/{market}"

    trades_request = {
        "req_id": f"{PING_ID}",
        "op": "subscribe",
        "args": [
            f"publicTrade.{symbol}",
            f"orderbook.1.{symbol}"
        ]}
    while True:
        try:
            async with websockets.connect(url) as websocket:
                await websocket.send(json.dumps(trades_request))
                # Receive and process incoming messages
                while True:
                    response = await websocket.recv()
                    response = json.loads(response)
                    if 'topic' in response.keys():
                        await data_handler.on_message(response)
        except websockets.exceptions.ConnectionClosedOK:
            print("WebSocket connection closed gracefully")
            await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosedError:
            print("WebSocket connection closed unexpectedly")
            await asyncio.sleep(1)
        except ConnectionResetError:
            print("Fucker got cancelled! It took me two weeks to isolate this error. Why GOD?")
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("Fucker got cancelled again! this time it was easier to pick up!!")
            await asyncio.sleep(1)
        except Exception as E:
            print("Exception occurred:", E)


async def gather_websocket_ping(symbol, market='spot'):
    # Run function_one and function_two concurrently
    await asyncio.gather(send_ping(market), websocket_client(symbol, market))


async def main():
    task = asyncio.create_task(gather_websocket_ping("BTCUSDT", "linear"))
    await task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")
