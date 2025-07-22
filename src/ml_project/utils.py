import os
import sys
from ml_project.exception import CustomException
from ml_project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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

def evaluate_models(x_train , y_train , x_test , y_test , models, param):

    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
            return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Loads the object from a pickle file at the specified file path.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)