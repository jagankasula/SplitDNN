import pickle
import tensorflow as tf
import tornado.ioloop

from tensorflow import keras
from vit_keras import vit


device = '/cpu:0'

with tf.device(device):
  model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=12,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)



split_point = 3
next_layer = model.layers[split_point + 1]

print(next_layer.name)

right_model = keras.Model(inputs=next_layer.input, outputs=model.output)


class ModelHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Model done")
    async def post(self):
        #data = json.loads(self.request.body)
            data =  pickle.loads(self.request.body)
            json_dump_return_data = model_right(data)
            self.write(json_dump_return_data)


def model_right(data):
   
    left_model_output = data['data']
    count = data['count']

    right_model_output = right_model(left_model_output)

    return_data = {'result':right_model_output, 'count':count}

    json_dump_return_data = pickle.dumps(return_data)

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