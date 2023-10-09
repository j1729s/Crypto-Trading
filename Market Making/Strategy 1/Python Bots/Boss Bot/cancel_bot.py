import time
import ccxt
import schedule
from time import sleep
from Secrets import API_KEY, API_SECRET
from threading import Thread


CHECK_LAG = 1  # In seconds

class CheckPosition(Thread):

    def __init__(self, symbol):
        Thread.__init__(self)

        self.symbol = symbol
        self.position = ""
        ## Exchange info and constants
        self.binance = ccxt.binanceusdm({'apiKey': API_KEY,
                                         'secret': API_SECRET})
        self.binance.load_markets()  # load markets to get the market id from a unified symbol

    def check_position(self):
        try:
            positions = self.binance.fetchPositions(symbols=[self.symbol], params={})[0]['info']['positionAmt']
            if positions != self.position:
                self.position = positions
                print("Position is: {}".format(positions))
        except Exception as e:
            print("Error while fetching positions")

    def run(self):
        schedule.every(0.5).seconds.do(self.check_position)
        while True:
            schedule.run_pending()
            sleep(0.5)

class CheckOrder_Panic(Thread):
    OPEN_ORDER_THRESHOLD = 50
    CANCEL_ORDER_THRESOLD = 5

    def __init__(self, symbol):
        Thread.__init__(self)

        self.symbol = symbol
        self.runtime = 0
        self.length = 0

        ## Exchange info and constants
        self.binance = ccxt.binanceusdm({'apiKey': API_KEY,
                                         'secret': API_SECRET})
        self.binance.load_markets()  # load markets to get the market id from a unified symbol

    def check_position(self):
        start = time.time()
        try:
            open_orders = self.binance.fetchOpenOrders(self.symbol)
            len_order = len(open_orders)
            print(f"Open Orders: {len_order}")
            if self.length != len_order:
                self.length = len_order
                print(f"Open Orders: {len_order}")
            if len_order > self.OPEN_ORDER_THRESHOLD:
                order_ids = [open_orders[i]["id"] for i in range(self.CANCEL_ORDER_THRESOLD)]
                print("Cancelling")
                for id in order_ids:
                    try:
                        self.binance.cancelOrder(id, symbol=self.symbol, params={})
                    except Exception as e:
                        print("Error while cancelling")
        except Exception as _:
            print(f"Exception while checking orders")
        end = time.time()
        self.runtime = round(end - start)

    def run(self):
        schedule.every(CHECK_LAG).seconds.do(self.check_position)
        while True:
            schedule.run_pending()
            sleep(CHECK_LAG - self.runtime)

def main():
    symbol = 'BTCBUSD'

    # Although threads are supposed to be functions we can use classes by overriding the Thread class of the Threading package
    # Refer: https://superfastpython.com/extend-thread-class/
    # Same is true for processes
    # Initialize each class
    task3 = CheckOrder_Panic(symbol)
    task4 = CheckPosition(symbol)

    # Start each one as a seperate thread
    task3.start()
    task4.start()

    # Join the threads
    task3.join()
    task4.join()

if __name__ == '__main__':
    main()