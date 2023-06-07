import datetime
from tornado.concurrent import Future
import concurrent
from tornado import gen, httpclient, ioloop, queues
import json
import numpy as np
from threading import *
from tornado.queues import Queue
from json import JSONEncoder
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import sys
import warnings
warnings.filterwarnings('ignore')

from transformers import ViTModel, ViTConfig

q = Queue(maxsize=2)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

device = 'cpu'

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
vit_model.config = vit_config

split_point = 768  # Split point for the transformer model

vit_model = vit_model.eval().to(device)


async def consumer():
    http_client = httpclient.AsyncHTTPClient(defaults=dict(connect_timeout=10000000.0, request_timeout=100000000000.0))
    count = 0

    startTime = datetime.datetime.now()

    async for item in q:
        try:
            count += 1
            item_np = item.detach().numpy()

            post_data = {'data': item_np, 'count': count}
            body = json.dumps(post_data, cls=NumpyArrayEncoder)
            a = sys.getsizeof(body)
            print(a)
            response = await http_client.fetch("http://localhost:8881/model", method='POST', headers=None, body=body)
            load_data = json.loads(response.body)
            result = load_data['result']
            count_return = load_data['count']
            result_right_np = np.asarray(result)
            result_right = torch.Tensor(result_right_np)
            
            with torch.no_grad():
                output = torch.nn.functional.softmax(result_right, dim=1)

            results = output.argmax(dim=1)
            print(f"Final Result for Image number {count}: {results}")
            endTime = datetime.datetime.now()
            print(f"Total processing time: {endTime - startTime}")

        finally:
            q.task_done()


def producer_video_left(img_rbg):
    count = 0

    img = Image.fromarray(img_rbg).convert('RGB')
    count += 1
    resize = transforms.Resize([224, 224])
    img = resize(img)

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        out_left = vit_model(tensor)[0]

    return out_left


async def main_runner():
    ioloop.IOLoop.current().add_callback(consumer)

    time_start = datetime.datetime.now()
    print('Timestamp Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    cam = cv2.VideoCapture('hdvideo.mp4')
    while True:
        ret, img_rbg = cam.read()
        if ret:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                out_left = await io_loop.run_in_executor(pool, producer_video_left, img_rbg)
            await q.put(out_left)
        else:
            time_finish = datetime.datetime.now()
            print('Timestamp Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            break
    print(f'Timestamp difference: {time_finish - time_start}')
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    io_loop = ioloop.IOLoop.current()
    io_loop.add_callback(main_runner)
    q.join()
    io_loop.start()