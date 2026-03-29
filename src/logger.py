'''
Logger provides a simple logging mechanism for the application. It logs every execution that happens in the application, making it easier to track the flow 
of the program and identify any issues that may arise. It allows for logging messages with different severity levels (INFO, WARNING, ERROR) and can be 
easily extended to include additional features such as file logging or log rotation.
'''

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)


'''-----Test code for logger-----
if __name__=='-_main__':
    logging.info("Logging has started")
    
'''

