import logging

class CustomLogger:
    def __init__(self, logger_path:str='/home/hjl/Chat_BDA/logs.txt'):       
        self.logger=logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(logger_path)
        # 挂载
        self.console_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
    
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