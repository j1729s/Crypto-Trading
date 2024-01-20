from Okx_Secret import API_KEY, API_SECRET, API_PASS
from multiprocessing import Process
import okx.Account as Account
import okx.Trade as Trade
from Constants import MARGIN_MODE, CONTRACT_SIZE, SYMBOL, GRID_SIZE
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
        self.bid_list = self.order_list[1: (GRID_SIZE + 1)]
        self.ask_list = self.order_list[(-1 - GRID_SIZE): -1]
        self.shared_orders = order_arr
        self.order = order_event

        ## Exchange info and constants
        self.Trade = Trade.TradeAPI(API_KEY, API_SECRET, API_PASS, False, "0")

    def send_orders(self):
        order_list = np.copy(np.frombuffer(self.shared_orders.get_obj(), dtype=np.float32))
        if order_list[0] == 0:
            if order_list[-1] == 1:
                new_bids = order_list[1:-1]
                new_asks = []
            elif order_list[-1] == -1:
                new_asks = order_list[1:-1]
                new_bids = []
        elif order_list[0] == 1:
            new_bids = order_list[1: (GRID_SIZE + 1)]
            new_asks = order_list[(-1 -GRID_SIZE): -1]
        bid_list = np.setdiff1d(new_bids, self.bid_list)
        self.bid_list = new_bids
        params_bid = [{'instId': SYMBOL, 'tdMode': MARGIN_MODE, 'side': 'buy', 'ordType': 'post_only', 'sz': CONTRACT_SIZE, 'posSide': 'net', 'px': str(px)} for px in bid_list]
        ask_list = np.setdiff1d(new_asks, self.ask_list)
        self.ask_list = new_asks
        params_ask = [{'instId': SYMBOL, 'tdMode': MARGIN_MODE, 'side': 'sell', 'ordType': 'post_only', 'sz': CONTRACT_SIZE, 'posSide': 'net', 'px': str(px)} for px in ask_list]
        params = params_bid + params_ask
        if len(params) > 0:
            with self.suppress_output():
                x = self.Trade.place_multiple_orders(params)
            if x['code'] == '0':
                print(f"{len(params)} Order sent!")
            else:
                print("Error while sending order: ", x['msg'])
                print(x['data'])



    def run(self):
        with self.order:
            while self.order.wait():
                self.send_orders()


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
        self.bid_list = self.order_list[1: (GRID_SIZE + 1)]
        self.ask_list = self.order_list[(-1 - GRID_SIZE): -1]
        self.shared_orders = order_arr
        self.cancel = cancel_event

        ## Exchange info and constants
        self.Trade = Trade.TradeAPI(API_KEY, API_SECRET, API_PASS, False, "0")

    def cancel_orders(self):
        order_list = np.copy(np.frombuffer(self.shared_orders.get_obj(), dtype=np.float32))
        if order_list[0] == 0:
            if order_list[-1] == 1:
                new_bids = order_list[1:-1]
                new_asks = []
            elif order_list[-1] == -1:
                new_asks = order_list[1:-1]
                new_bids = []
        elif order_list[0] == 1:
            new_bids = order_list[1: (GRID_SIZE + 1)]
            new_asks = order_list[(-1 - GRID_SIZE): -1]
        bid_list = np.setdiff1d(self.bid_list, new_bids)
        self.bid_list = new_bids
        ask_list = np.setdiff1d(self.ask_list, new_asks)
        self.ask_list = new_asks
        cancel_list = np.append(bid_list, ask_list)
        with self.suppress_output():
            x = self.Trade.get_order_list(instId="BTC-USDT-SWAP", ordType='post_only')
        data = x['data']
        if len(data) > 0 and len(cancel_list) > 0:
            params = [{'instId': "BTC-USDT-SWAP", 'ordId': dat['ordId']} for dat in data
                      if float(dat['px']) in cancel_list]
            if len(params) > 0:
                with self.suppress_output():
                    x = self.Trade.cancel_multiple_orders(params)
                if x['code'] == '0':
                    print(f"{len(params)} Orders Cancelled")
                else:
                    print("Error while cancelling: ", x['msg'])

    def run(self):
        with self.cancel:
            while self.cancel.wait():
                self.cancel_orders()