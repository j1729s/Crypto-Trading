import asyncio
import numpy as np
from Bybit_Constants import *
from pybit.unified_trading import HTTP
from ByBit_Secret import API_KEY, API_SECRET

# Set up logging (optional)
import logging
logging.basicConfig(filename="pybit.log", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")


class CircularBuffer:
    def __init__(self, max_rows, num_columns):
        self.max_rows = max_rows
        self.num_columns = num_columns
        self.buffer = np.zeros((max_rows, num_columns))
        self.start = 0
        self.count = 0

    def append(self, values):
        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                values = values.reshape(1, -1)
            elif values.ndim != 2 or values.shape[1] != self.num_columns:
                raise ValueError(f"Each array must have shape ({self.num_columns},)")
        else:
            values = np.array(values)
            if values.ndim != 2 or values.shape[1] != self.num_columns:
                raise ValueError(f"Each array must have shape ({self.num_columns},)")

        for value in values:
            self.buffer[self.start] = value
            self.start = (self.start + 1) % self.max_rows
            if self.count < self.max_rows:
                self.count += 1

    def get(self):
        if self.count == 0:
            return np.array([])  # buffer is empty
        if self.count == self.max_rows:
            indices = (np.arange(self.start, self.start + self.max_rows) % self.max_rows)
        else:
            indices = np.arange(0, self.count)
        return self.buffer[indices]

    def __str__(self):
        return str(self.get())


class OrderPlacement:
    def __init__(self):
        self.quant = MIN_ORD_QUANT * DELTA
        self.session = HTTP(testnet=False,
                            demo=True,
                            api_key=API_KEY,
                            api_secret=API_SECRET)
        print("OrderPlacement Class initiated!")

    async def place_order(self, price_array):
        ord_request = [
                {
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Limit",
                    "qty": f"{self.quant}",
                    "price": f"{price_array[0]}",
                    "timeInForce": "PostOnly",
                },
                {
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "orderType": "Limit",
                    "qty": f"{self.quant}",
                    "price": f"{price_array[1]}",
                    "timeInForce": "PostOnly",
                },
            ]
        print("Placing Order: ", price_array)
        order = self.session.place_batch_order(category="linear",
                                               request=ord_request)
        if order['retMsg'] == 'OK':
            print("Order Placed successfully")
            open_orders = self.session.get_open_orders(category="linear",
                                                       symbol="BTCUSDT",
                                                       openOnly=0)
            if len(open_orders['result']['list']) != 2:
                return True   ## For retrying
        else:
            print(f"Exception occurred during order placement: {order}")

    async def place_custom_order(self, price, quant, side):
        print(f"Placing Custom Order at {price} for {quant} on {'long' if side == 1 else 'short'}")
        sd = "Sell" if side == -1 else "Buy"
        order = self.session.place_order(category="linear",
                                         symbol="BTCUSDT",
                                         side=sd,
                                         orderType="Limit",
                                         qty=f"{quant}",
                                         price=f"{price}",
                                         timeInForce="PostOnly")
        if order['retMsg'] == 'OK':
            print("Custom Order Placed successfully")
        else:
            print(f"Exception occurred during custom order placement: {order}")

    async def cancel_orders(self):
        print("Cancelling all USDT Orders!!")
        cancel = self.session.cancel_all_orders(category="linear",
                                                settleCoin="USDT",)
        if cancel['retMsg'] == 'OK':
            print("Orders cancelled successfully")
        else:
            print(f"Exception occurred during order cancellation: {cancel}")

class GetPosition:
    def __init__(self):
        self.session = HTTP(testnet=False,
                            demo=True,
                            api_key=API_KEY,
                            api_secret=API_SECRET)
        print("GetPosition class initiated!")

    async def get_positions(self):
        position_dict = self.session.get_positions(category="linear",
                                               symbol="BTCUSDT")
        if position_dict['retMsg'] == 'OK':
            return position_dict
        else:
            print(f"Exception while getting position: {position_dict}")


if __name__ == "__main__":
    op = GetPosition()
    print(asyncio.run(op.get_positions()))