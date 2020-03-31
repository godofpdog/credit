import os
import time
import logging
import datetime

def mkdir(path):
    path.strip()
    path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

class Logger():
    def __init__(self, name = "Logger", log_dir = 'logs/'):
        super().__init__()
        self.name = name
        self.log_dir = log_dir
        self.another_day = False
        self.initial_logger()
        

    def initial_logger(self):

        self.now_time = datetime.datetime.now()

        log_file_root_path = os.path.join(project_root_dir.project_dir, self.log_dir)
        log_time = time.strftime('%Y_%m_%d', time.localtime(time.time()))
        path_join = os.path.join(log_file_root_path, self.name)
        mkdir(path_join)
        
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

        log_file = os.path.join(path_join, '{}.log'.format(log_time)) 

        if self.another_day:
            self.logger.handlers = []

        if len(self.logger.handlers) == 0:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s -  %(threadName)s - %(process)d ")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)
            self.another_day = False
        self.logger.info('logger initialize at {}'.format(self.now_time))
        return None

    def check_time(self):
        delta = datetime.timedelta(days=1)
        # delta = datetime.timedelta(minutes=1)
        now_time = int(datetime.datetime.now().strftime('%d'))
        diff = self.now_time - delta
        diff = int(diff.strftime('%d'))
        if now_time - diff > 1:
            self.another_day = True
            self.initial_logger()
        return None

    def debug(self, msg, *args, **kwargs):
        # self.check_time()
        if self.logger is not None:
            self.logger.debug(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        # self.check_time()
        if self.logger is not None:
            self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        # self.check_time()
        if self.logger is not None:
            self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        # self.check_time()
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        # self.check_time()
        if self.logger is not None:
            self.logger.warning(msg, *args, exc_info=True, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        # self.check_time()
        if self.logger is not None:
            self.logger.exception(msg, *args, exc_info=True, **kwargs)
