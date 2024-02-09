import os
import time
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

class Logger():
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.save_path = f"{time.time()}"

    def init(self):
        if not os.path.exists("logs"):
            os.mkdir("logs")
        os.mkdir("logs"+"/"+self.save_path)

    def log(self, loss, train=True):
        if train:
            self.train_loss.append(loss)
        else:
            self.eval_loss.append(loss)            

    def logging(self, log):
        san_francisco_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.utcnow()
        sf_time = now.astimezone(san_francisco_tz)
        formatted_date = sf_time.strftime("%m-%d %H:%M")
        
        text = f'{formatted_date} : {log}\n'
        with open(f'logs/{self.save_path}/logs.txt', 'a') as file:
            file.write(text)

    def draw_loss(self, train=True):
        if train:
            plt.plot(self.train_loss)
            plt.savefig(f'logs/{self.save_path}/train_log.png')
            plt.close()
        else:
            plt.plot(self.eval_loss)
            plt.savefig(f'logs/{self.save_path}/valid_log.png')
            plt.close()
