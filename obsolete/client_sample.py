import pickle
import tornado.httpclient
import tornado.ioloop
import cv2
import datetime


def send_image_to_server(input_path):
    #
    url = "http://localhost:8888/upload_data"
    print("connected to server")
    http_client = tornado.httpclient.HTTPClient()

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_length = total_frames / fps
    print(f"No of frames: {total_frames}")
    print(f"Video length: {video_length:.2f} seconds")

    startTime = datetime.datetime.now()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        try:
            data = pickle.dumps(frame)
            request = tornado.httpclient.HTTPRequest(url=url, method='POST', body=data)
            response = http_client.fetch(request)
        except tornado.httpclient.HTTPError as e:
            print(f"HTTP Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
    http_client.close()
    endTime = datetime.datetime.now()
    print(f"total processing time: {endTime - startTime}")


if __name__ == '__main__':
    send_image_to_server('hdvideo.mp4')

# Load the car cascade classifier
car_cascade = cv2.CascadeClassifier('cars.xml')
if car_cascade.empty():
    raise IOError('Unable to load the car cascade classifier xml file')
