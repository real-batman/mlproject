import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Train test split initiated')
            train_set,test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occurred in data ingestion component")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()


#---- IGNORE THIS ----

'''
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

def get_data_ingestion_config():
    """Returns a dictionary of paths used for data ingestion."""
    return {
        "train_data_path": os.path.join('artifacts', 'train.csv'),
        "test_data_path": os.path.join('artifacts', 'test.csv'),
        "raw_data_path": os.path.join('artifacts', 'data.csv')
    }

def initiate_data_ingestion():
    """Reads, splits, and saves the dataset to the artifacts directory."""
    logging.info("Entered the data ingestion function")
    
    try:
        # Load configuration
        config = get_data_ingestion_config()
        
        # Read the dataset
        df = pd.read_csv('notebook/data/stud.csv')
        logging.info('Read the dataset as dataframe')

        # Create the artifacts directory if it doesn't exist
        os.makedirs(os.path.dirname(config['raw_data_path']), exist_ok=True)

        # Save the raw data first
        df.to_csv(config['raw_data_path'], index=False, header=True)

        logging.info('Train test split initiated')
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        # Save train and test sets
        train_set.to_csv(config['train_data_path'], index=False, header=True)
        test_set.to_csv(config['test_data_path'], index=False, header=True)

        logging.info("Ingestion of the data is completed")

        return (
            config['train_data_path'],
            config['test_data_path']
        )

    except Exception as e:
        logging.error("Error occurred in data ingestion function")
        raise CustomException(e, sys)

# To execute the process:
# if __name__ == "__main__":
#     train_path, test_path = initiate_data_ingestion()
'''