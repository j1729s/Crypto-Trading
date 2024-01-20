from Okx_Secret import API_KEY, API_SECRET, API_PASS
from multiprocessing import Process
import okx.Account as Account
import okx.Trade as Trade
from Constants_GLFT import MARGIN_MODE, CONTRACT_SIZE, SYMBOL
import numpy as np
import sys
from contextlib import contextmanager

class CheckPosition(Process):
    @contextmanager
    def suppress_output(self):
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')  # Redirect to /dev/null or 'nul' on Windows
        try:
            yield
        finally:
            sys.stdout = original_stdout

    def __init__(self, value, event):
        Process.__init__(self)
        self.Account = Account.AccountAPI(API_KEY, API_SECRET, API_PASS, False, '0')
        with self.suppress_output():
            self.Account.set_leverage(instId="BTC-USDT-SWAP",
                                      lever="100",
                                      mgnMode=MARGIN_MODE,
                                      posSide='net')
        self.value = value
        self.event = event
        self.pos = 0

    def check_position(self):
        try:
            with self.suppress_output():
                x = self.Account.get_positions(instId='BTC-USDT-SWAP')
            if x['code'] == '0':
                data = x['data']
                for dat in data:
                    if dat['mgnMode'] == MARGIN_MODE:
                        self.pos = int(dat['pos'])
                        self.value.value = self.pos
            else:
                print("Error while getting position: ", x['msg'])
                self.value.value = self.pos
        except Exception as E:
            print(f"Exception while fetching position: {E}")
            self.value.value = self.pos

    def run(self):
        with self.event:
            while self.event.wait():
                self.check_position()


class OrderPlacement(Process):
    @contextmanager
    def suppress_output(self):
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')  # Redirect to /dev/null or 'nul' on Windows
        try:
            yield
        finally:
            sys.stdout = original_stdout

    def __init__(self, order_arr, order_event):
        Process.__init__(self)
        ## Trading Metrics
        self.symbol = SYMBOL
        self.size = CONTRACT_SIZE
        self.order_list = np.copy(np.frombuffer(order_arr.get_obj(), dtype=np.float32))
        self.bid_list = self.order_list[0:1]
        self.ask_list = self.order_list[1:]
        self.shared_orders = order_arr
        self.order = order_event

        ## Exchange info and constants
        self.Trade = Trade.TradeAPI(API_KEY, API_SECRET, API_PASS, False, "0")

    @staticmethod
    def count_unique(arr):
        # Count the occurrences of each unique element
        unique_elements, counts = np.unique(arr, return_counts=True)
        # Create a dictionary to store the counts for each element
        element_counts = dict(zip(unique_elements, counts))

        return element_counts

    def gen_order_list(self, side, new_side, side_list):
        side_unique = self.count_unique(new_side)
        side_count_dict = {key: value for key, value in side_unique.items() if key in side_list}

        return [{'instId': SYMBOL, 'tdMode': MARGIN_MODE, 'side': side, 'ordType': 'post_only', 'sz': CONTRACT_SIZE,
                'posSide': 'net', 'px': str(key)} for key, value in side_count_dict.items() for _ in range(value)]

    def create_order_params(self):
        order_list = np.copy(np.frombuffer(self.shared_orders.get_obj(), dtype=np.float32))
        new_bids = order_list[0:1]
        new_asks = order_list[1:]
        bid_list = np.setdiff1d(new_bids, self.bid_list)
        self.bid_list = new_bids
        self.gen_order_list('buy', new_bids, bid_list)
        ask_list = np.setdiff1d(new_asks, self.ask_list)
        self.ask_list = new_asks
        params_bid = self.gen_order_list('buy', new_bids, bid_list)
        params_ask = self.gen_order_list('sell', new_asks, ask_list)
        print("New Bids and Asks are: ", new_bids, new_asks)
        return params_bid, params_ask

    def send_orders(self, bids, asks):
        params = bids + asks
        if len(params) > 0:
            with self.suppress_output():
                x = self.Trade.place_multiple_orders(params)
            if x['code'] == '0':
                print(f"{len(bids)} Buy Orders and {len(asks)} Sell Orders sent!")
            else:
                print("Error while sending order: ", x['msg'])
                print(x['data'])

    def run(self):
        with self.order:
            while self.order.wait():
                bids, asks = self.create_order_params()
                try:
                    self.send_orders(bids, asks)
                except Exception as E:
                    print("Exception while placing order ", E)
                    print("Trying again!")
                    try:
                        self.send_orders(bids, asks)
                    except Exception as e:
                        print("Exception occurred upon retrial of order ", e)


class CancelOrders(Process):
    @contextmanager
    def suppress_output(self):
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')  # Redirect to /dev/null or 'nul' on Windows
        try:
            yield
        finally:
            sys.stdout = original_stdout

    def __init__(self, order_arr, cancel_event):
        Process.__init__(self)
        self.order_list = np.copy(np.frombuffer(order_arr.get_obj(), dtype=np.float32))
        self.bid_list = self.order_list[0:1]
        self.ask_list = self.order_list[1:]
        self.shared_orders = order_arr
        self.cancel = cancel_event

        ## Exchange info and constants
        self.Trade = Trade.TradeAPI(API_KEY, API_SECRET, API_PASS, False, "0")

    def create_order_list(self):
        order_list = np.copy(np.frombuffer(self.shared_orders.get_obj(), dtype=np.float32))
        self.bid_list = order_list[0:1]
        self.ask_list = order_list[1:]
        return {'buy': self.bid_list, 'sell': self.ask_list}

    def cancel_orders(self, order_list):
        with self.suppress_output():
            x = self.Trade.get_order_list(instId="BTC-USDT-SWAP", ordType='post_only')
        data = x['data']
        if len(data) > 0:
            params_bid = [{'instId': "BTC-USDT-SWAP", 'ordId': dat['ordId']} for dat in data
                          if float(dat['px']) not in order_list['buy'] and dat['side'] == 'buy']
            params_ask = [{'instId': "BTC-USDT-SWAP", 'ordId': dat['ordId']} for dat in data
                          if float(dat['px']) not in order_list['sell'] and dat['side'] == 'sell']
            params = params_bid + params_ask
            if len(params) > 0:
                with self.suppress_output():
                    x = self.Trade.cancel_multiple_orders(params[:20])
                if x['code'] == '0':
                    print(f"{len(params)} Orders Cancelled")
                else:
                    print("Error while cancelling: ", x['msg'])


    def run(self):
        with self.cancel:
            while self.cancel.wait():
                order_list = self.create_order_list()
                try:
                    self.cancel_orders(order_list)
                except Exception as E:
                    print("Exception while cancelling orders ", E)
                    print("Trying again")
                    try:
                        self.cancel_orders(order_list)
                    except Exception as e:
                        print("Exception occurred upon retrial of cancellation ", e)