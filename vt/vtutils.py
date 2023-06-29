import json
import datetime

class Config:
    @staticmethod
    def get_config():
        with open('config.json') as config_file:
            config = json.load(config_file)
            config['url'] = config['url'].replace("{{server}}", config['server'])
        return config


class Logger():
    def log(message):
        time = datetime.datetime.now()
        print(f"{time}::{message}")