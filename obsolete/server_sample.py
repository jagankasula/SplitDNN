import pickle

import tornado.ioloop
import tornado.web
import cv2
import datetime
import numpy as np


class DataHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            input_data = self.request.body
            print(f"Received frame with {len(input_data)} bytes")
        except Exception as e:
            print(f"Error: {e}")
        processInput(input_data)
        self.write("Frame received by the server")


def processInput(input_data):
    startTime = datetime.datetime.now()
    frame = pickle.loads(input_data)
    frame = detect_cars(frame)
    # cv2.imshow('frame', frame)
    endTime = datetime.datetime.now()
    print("Computation time: ", endTime - startTime)
    cv2.destroyAllWindows()


def detect_cars(frame):
    # print("Entered frame detection")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame


car_cascade = cv2.CascadeClassifier('cars.xml')
if car_cascade.empty():
    raise IOError('Unable to load the car cascade classifier xml file')


if __name__ == '__main__':
    app = tornado.web.Application([
        (r"/upload_data", DataHandler),
    ])
    app.listen(8888)
    print("Server started and listening...")
    tornado.ioloop.IOLoop.current().start()
