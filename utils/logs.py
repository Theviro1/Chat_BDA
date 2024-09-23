import logging
import sys
from utils.config import *

class BaseLogger:
    def __init__(self, logger_path:str, logger_name:str):               
        self.logger=logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # delete the protential repeat handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            file_handler = logging.FileHandler(logger_path, mode='w', encoding='utf-8')
            console_handler.setFormatter(self.formatter)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(e)
    
    def warning(self, msg:str):
        self.logger.warning(msg)
    
    def info(self, msg:str):
        self.logger.info(msg)

    def error(self, msg:str):
        self.logger.error(msg)
    
    def debug(self, msg:str):
        self.logger.debug(msg)
    
    def critical(self, msg:str):
        self.logger.critical(msg)

class RAGLogger(BaseLogger):
    def __init__(self):
        super().__init__(logger_path=RAG_LOGGER_PATH, logger_name='RAG')

class FreeMODLogger(BaseLogger):
    def __init__(self):
        super().__init__(logger_path=FREEMOD_LOGGER_PATH, logger_name='FreeMOD')

class ChatIELogger(BaseLogger):
    def __init__(self):
        super().__init__(logger_path=CHATIE_LOGGER_PATH, logger_name='ChatIE')
