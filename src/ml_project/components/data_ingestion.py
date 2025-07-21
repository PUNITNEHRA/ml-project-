# database ---> data --> trin test split
import os
import sys
from ml_project.exception import CustomException
from ml_project.logger import logging
import pandas as pd
from ml_project.utils import read_sql_data

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    logging.info("Data Ingestion Config Initialized")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    logging.info("Data Ingestion Config ended")
    
class DataIngestion:
    logging.info("Data Ingestion Class Initialized")
    def __init__(self):
        self.ingestion_config = DataIngestionConfig
    logging.info("Data Ingestion Class ENDED")
    



    def initiate_data_ingestion(self):
        try:

            # df = read_sql_data()
            df = pd.read_csv(os.path.join('notebook/data' , 'raw.csv'))
            logging.info("reading completed data from SQL database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False , header=True)
            train_set , test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False , header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False , header=True)

            logging.info("data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e , sys)

