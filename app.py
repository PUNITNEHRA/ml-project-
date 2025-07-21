import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from ml_project.components.data_ingestion import DataIngestionConfig, DataIngestion     
from ml_project.components.data_transformation import DataTransformationConfig, DataTransformation     


from ml_project.logger import logging
from ml_project.exception import CustomException
logging.info("Logger test successful!")

try:
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path, test_data_path)

except ZeroDivisionError as e:
    logging.error(f"Error occurred: {e}")
    raise CustomException("A custom error occurred", error_code=500)    
