import os
import sys
from ml_project.exception import CustomException
from ml_project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

import pickle
import numpy as np

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading data from SQL database")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("connection established" , mydb)
        df = pd.read_sql_query('Select * from students', mydb)
        print(df.head())
        return df
    
    except Exception as e:
        raise CustomException(e)
    
def save_object(file_path, obj):
    """
    Saves the object as a pickle file at the specified file path.
    """
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved at {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)       