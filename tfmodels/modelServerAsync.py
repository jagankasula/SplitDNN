import pickle
import datetime
import tensorflow as tf
import tornado.ioloop

from tensorflow import keras
from modelUtils import Config, Logger, get_model, dry_run_right_model


# Read the configurations from the config file.
config = Config.get_config()

device = config['server_device']
model_name = config['model']
split_point = None

with tf.device(device):
  model = get_model(model_name)
  print('*************************************************')
  print(tf.config.list_physical_devices(device_type=None))
  print('**************************************************')

  #right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
  left_model = None
  right_model = None
  total_right_model_time = 0


class ModelHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Model done")
    def post(self):
        #data = json.loads(self.request.body)
        server_request_receive_timestamp = datetime.datetime.now()
        data =  pickle.loads(self.request.body)
        return_data = model_right(data)
        # server_processing_timestamp = datetime.datetime.now()
        # return_data['server_processing_time'] = (server_processing_timestamp - server_request_receive_timestamp).total_seconds()
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

class TimeHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Model done")
    def post(self):
        #data = json.loads(self.request.body)
        data =  pickle.loads(self.request.body)
        json_dump_return_data = get_total_right_model_time(data)
        self.write(json_dump_return_data)

def set_model_right(data):    
    global right_model
    global left_model
    global split_point
    split_point = data['split_point']
    split_layer = model.layers[split_point]
    left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
    next_layer = model.layers[split_point + 1]
    print(f'Starting from layer # {split_point + 1} in server. Layer name: {next_layer.name}')
    right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
    dry_run_right_model(left_model, right_model)
    return pickle.dumps('Right model is ready.')

def model_right(data):   
   with tf.device(device):       
       left_model_output = data['data']
       frame_seq_no = data['frame_seq_no']
       Logger.log(f'Executing right model for frame #: {frame_seq_no}')
       right_model_start_time = datetime.datetime.now()
       right_model_output = right_model(left_model_output)
       right_model_end_time = datetime.datetime.now()
       add_to_total_right_model_time((right_model_end_time - right_model_start_time).total_seconds())
       return_data = {'result':right_model_output, 'frame_seq_no':frame_seq_no}       
       return return_data
   
def get_total_right_model_time(data):
    global total_right_model_time
    client_split_point = data['split_point']
    return_data = None
    if client_split_point == split_point:
        return_data = {'total_right_model_time': total_right_model_time}
        # Reset time for next split point.
        total_right_model_time = 0        
    else:
        return_data = {'total_right_model_time': 'split point mismatch'}
    return pickle.dumps(return_data)


def add_to_total_right_model_time(current_frame_exec_time):
   global total_right_model_time
   total_right_model_time += current_frame_exec_time      
     

def make_app():
    return tornado.web.Application([
        (r"/model", ModelHandler),
        (r"/split_point", SplitPointHandler),
        (r"/right_model_time", TimeHandler)
    ])

with tf.device(device):
   if __name__ == "__main__":
        app = make_app()
        app.listen(8881)
        print("Server started")
        tornado.ioloop.IOLoop.current().start()     