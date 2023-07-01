import pickle
import datetime
import tensorflow as tf
import tornado.ioloop

from tensorflow import keras
from vit_keras import vit
from vtutils import Config, Logger


# Read the configurations from the config file.
config = Config.get_config()

device = config['server_device']

split_point = None

with tf.device(device):
    model = vit.build_model(image_size=224, patch_size=16, classes=1000, num_layers=19,
                        hidden_size=768, num_heads=12, name= 'vit_custom', mlp_dim=3072,
                        activation='softmax', include_top=True,
                        representation_size=None)

    #right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
    right_model = None


class ModelHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Model done")
    def post(self):
        #data = json.loads(self.request.body)
            server_request_receive_timestamp = datetime.datetime.now()
            data =  pickle.loads(self.request.body)
            return_data = model_right(data)
            server_processing_timestamp = datetime.datetime.now()
            return_data['server_processing_time'] = server_processing_timestamp - server_request_receive_timestamp
            json_dump_return_data = pickle.dumps(return_data)
            self.write(json_dump_return_data)


class SplitPointHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Model done")
    def post(self):
        #data = json.loads(self.request.body)
            data =  pickle.loads(self.request.body)
            json_dump_return_data = set_model_right(data)
            self.write(json_dump_return_data)

def set_model_right(data):
     
     global right_model
     global split_point

     split_point = data['split_point']

     next_layer = model.layers[split_point + 1]

     print(f'Starting from layer # {split_point + 1} in server. Layer name: {next_layer.name}')

     right_model = keras.Model(inputs=next_layer.input, outputs=model.output)

     return pickle.dumps('Right model is ready.')


def model_right(data):
   
   with tf.device(device):
       
       left_model_output = data['data']

        # Transformer layers start from layer 5 and the output of the layer is two tensors. But the next layer input is only first tensor.            
       if split_point >= 5:
            left_model_output = left_model_output[0]

       frame_seq_no = data['frame_seq_no']

       Logger.log(f'Executing right model for frame #: {frame_seq_no}')

       right_model_output = right_model(left_model_output)

       return_data = {'result':right_model_output, 'frame_seq_no':frame_seq_no}       

       return return_data       
     

def make_app():
    return tornado.web.Application([
        (r"/model", ModelHandler),
        (r"/split_point", SplitPointHandler)
    ])

with tf.device(device):
     if __name__ == "__main__":
        app = make_app()
        app.listen(8881)
        print("Server started")
        tornado.ioloop.IOLoop.current().start()