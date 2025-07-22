from dagshub import dagshub_logger, init
import mlflow

init(repo_owner='PUNITNEHRA', repo_name='ml-project-', mlflow=True)


import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from ml_project.components.data_ingestion import DataIngestionConfig, DataIngestion     
from ml_project.components.data_transformation import DataTransformationConfig, DataTransformation     


from ml_project.logger import logging
from ml_project.exception import CustomException

from ml_project.components.model_trainer import ModelTrainerConfig, ModelTrainer

logging.info("Logger test successful!")

try:
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation()
    train_arr , test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    # model_trainer_config = ModelTrainerConfig()
    logging.info("Model Trainer Config Initialized")
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

except ZeroDivisionError as e:
    logging.error(f"Error occurred: {e}")
    raise CustomException("A custom error occurred", error_code=500)    
