import tornado.ioloop
import tornado.websocket
import time

class WebSocketClient(tornado.websocket.WebSocketClientConnection):
    def __init__(self, url, *args, **kwargs):
        super().__init__(url, *args, **kwargs)
        self.latency = None

    def on_message(self, message):
        if message == "Pong":
            self.latency = time.time() - self.start_time
            print(f"Latency: {self.latency:.6f} seconds")
            self.close()

    def on_close(self):
        tornado.ioloop.IOLoop.current().stop()

def connect_to_server():
    url = "ws://localhost:8881/websocket"
    client = tornado.websocket.websocket_connect(url)
    client.start_time = time.time()
    return client

if __name__ == "__main__":
    client = connect_to_server()
    tornado.ioloop.IOLoop.current().start()
