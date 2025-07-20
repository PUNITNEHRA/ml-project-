import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from ml_project.components.data_ingestion import DataIngestionConfig, DataIngestion     

from ml_project.logger import logging
from ml_project.exception import CustomException
logging.info("Logger test successful!")

try:
    logging.info("Data Ingestion Config Initialized")
    data_ingestion = DataIngestion()
    logging.info("Data Ingestion Config ended")
    logging.info("Data Ingestion Class Initialized-----------")
    data_ingestion.initiate_data_ingestion()
    logging.info("Data Ingestion Class ended")
except ZeroDivisionError as e:
    logging.error(f"Error occurred: {e}")
    raise CustomException("A custom error occurred", error_code=500)    
