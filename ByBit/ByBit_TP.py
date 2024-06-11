import json
import asyncio
import websockets
from Bybit_Constants import TP_INTERVAL
from Bybit_GLFT_Helper import OrderPlacement, GetPosition

PING_ID = 970479

class TakeProfits:

    def __init__(self):
        self.upl = 0.0
        self.mid = 0.0
        self.side = 0.0
        self.position = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.avg_price = 0.0
        self.mark_price = 0.0
        self._mid_lock = asyncio.Lock()
        self.position_system = GetPosition()
        self.order_system = OrderPlacement()

    async def handle_level1(self, data):
        async with self._mid_lock:
            if len(data['b']) > 0:
                self.best_bid = float(data['b'][0][0])
            if len(data['a']) > 0:
                self.best_ask = float(data['a'][0][0])
            self.mid = round((self.best_bid + self.best_ask) / 2, 2)

    async def catch_profits(self):
        async with self._mid_lock:
            if (self.side == -1 and self.mid < self.avg_price) or (self.side == 1 and self.mid > self.avg_price):
                print(f"Catching Profit at Position = {self.position}")
                print(f"Unrealised PnL is {self.upl}", self.mid, self.avg_price)
                quant = abs(self.position)
                side = int(self.side * -1)
                price = self.best_ask + 1 if self.side == 1 else self.best_bid - 1
                await self.order_system.cancel_orders()
                await self.order_system.place_custom_order(price, quant, side)

    async def maintain_position(self):
        print("Maintaining Position")
        while True:
            position_dict = await self.position_system.get_positions()
            position = position_dict['result']['list'][0]['size']
            upl = position_dict['result']['list'][0]['unrealisedPnl']
            avg_price = position_dict['result']['list'][0]['avgPrice']
            mark_price = position_dict['result']['list'][0]['markPrice']
            mark_price = float(mark_price) if bool(mark_price) else 0.0
            async with self._mid_lock:
                self.upl = float(upl) if bool(upl) else 0.0
                self.avg_price = float(avg_price) if bool(avg_price) else self.avg_price
                self.side = 1 if position_dict['result']['list'][0]['side'] == 'Buy' else -1
                self.position = float(position) * self.side if bool(position) else 0.0
            print([mark_price, self.mark_price, self.position, self.upl])
            if mark_price != self.mark_price:
                self.mark_price = mark_price
                if self.position:
                    await self.catch_profits()
            await asyncio.sleep(TP_INTERVAL)

    async def send_ping(self, market='spot'):
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
    async def websocket_client(self, symbol, market='spot'):
        url = f"wss://stream.bybit.com/v5/public/{market}"

        trades_request = {
            "req_id": f"{PING_ID}",
            "op": "subscribe",
            "args": [
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
                            await self.handle_level1(response['data'])
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

    async def gather_websocket_ping(self, symbol, market='spot'):
        # Run function_one and function_two concurrently
        await asyncio.gather(self.send_ping(market), self.websocket_client(symbol, market))


async def main():
    tp = TakeProfits()
    task = asyncio.create_task(tp.gather_websocket_ping("BTCUSDT", "linear"))
    await asyncio.gather(tp.gather_websocket_ping("BTCUSDT", "linear"), tp.maintain_position())
    #await task

if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Stopping...")
        except Exception as E:
            print("Exception occurred: ", E)