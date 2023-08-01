import tornado.ioloop
import tornado.web
import tornado.websocket
import time

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print("WebSocket connection established.")

    def on_message(self, message):
        current_time = time.time()
        self.write_message("Pong")
        latency = time.time() - current_time
        print(f"Latency: {latency:.6f} seconds")

    def on_close(self):
        print("WebSocket connection closed.")

def make_app():
    return tornado.web.Application([(r"/websocket", WebSocketHandler)])

if __name__ == "__main__":
    app = make_app()
    app.listen(8881)
    print("Server listening on port 8881")
    tornado.ioloop.IOLoop.current().start()
