import concurrent
import json
import numpy as np
import time
import torch
import torch.nn as nn
import tornado.ioloop
import tornado.web
import warnings

from tornado import gen, httpclient, ioloop, queues
from torch.nn.modules.container import Sequential
from utils import Config, Flatten, Logger, NumpyArrayEncoder, my_models



warnings.filterwarnings('ignore')

config = Config.get_config()

split_point = config['split_point']
device = config['server_device']

model_name = config['model']
my_model = my_models.get(model_name, lambda: print(f"Model not present in my_models: {model_name}"))(pretrained=True)

sub_models = [my_model]

for sub_model in sub_models:
    all_modules = []
    for module in sub_model.children():
        if isinstance(module,Sequential):
            for m in module.children():
                all_modules.append(m)

        else:
            all_modules.append(module)
    
    
all_modules = nn.Sequential(*all_modules)
right_modules = list(all_modules.children())[split_point:-1]
resnet_right = nn.Sequential(*[*right_modules, Flatten(), list(all_modules.children())[-1]])

resnet_right = resnet_right.eval().to(device)


io_loop = ioloop.IOLoop.current()

class ModelHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Model done")
    async def post(self):
        #data = json.loads(self.request.body)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            data = await io_loop.run_in_executor(pool, json.loads, self.request.body)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            json_dump_return_data = await io_loop.run_in_executor(pool, model_right, data)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await io_loop.run_in_executor(pool, self.write, json_dump_return_data)

def model_right(data):
    #time.sleep(2)    
    x = data['data']
    count = data['count']
    Logger.log(f"processing {count}")
    #print(f"Receiving data from client: {count}")
    result_left_np=np.asarray(x)
    result_left = torch.Tensor(result_left_np)
    #print(result_left)
    print(f"Sending data to the right side of the model: {count}")


    out_right=resnet_right(result_left)
    out_right=out_right.cpu()
    #log(f"Receiving data from right side of the model: {count}")
    result_right_np = out_right.detach().numpy()
    return_data = {'result':result_right_np, 'count':count}
    Logger.log(f"Transferring data back to the client: {count}")
    json_dump_return_data = json.dumps(return_data, cls=NumpyArrayEncoder)

    #print(out_right)
    return json_dump_return_data



def make_app():
    return tornado.web.Application([
        (r"/model", ModelHandler)
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8881)
    print("Got a Call")
    tornado.ioloop.IOLoop.current().start()