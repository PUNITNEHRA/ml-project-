import os      
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

from catboost import CatBoostRegressor
import mlflow
import mlflow.sklearn



import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ml_project.exception import CustomException
from ml_project.logger import logging
from ml_project.utils import save_object, evaluate_models





@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def eval_metrics(self, actual, pred):
        """
        Evaluates the model's performance using RMSE, MAE, and R2 score.
        """
        rmse = np.sqrt(np.mean(mean_squared_error(actual, pred)))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        logging.info(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        
        return rmse, mae, r2
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },     
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.5, 0.001],
                    'subsample': [0.5, 0.7, 0.9, 1.0],
                    'n_estimators': [50, 100, 200],
                },
                "Linear Regression": {},
                "XGB Regressor": {
                    'learning_rate': [0.01, 0.1, 0.5, 0.001],
                    'n_estimators': [50, 100, 200],
                },
                "CatBoost Regressor": {
                    'learning_rate': [0.01, 0.1, 0.5, 0.001],
                    'depth': [4, 6, 8, 10],
                    'iterations': [50, 100, 200],    
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 0.001],
                },
            }


            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            
            print("this is  the best model", best_model_name)
            
            model_names =  list(params.keys())
            
            actual_model = ""
            
            for model in model_names:
                if model == best_model_name:
                    actual_model = actual_model + model
                    break
            
            best_params = params[actual_model]
            
            mlflow.set_registry_uri("https://dagshub.com/PUNITNEHRA/ml-project-.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            #ml flow
            
            with mlflow.start_run():
                
                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                # mlflow.log_param(best_params)
                for key, value in best_params.items():
                    mlflow.log_param(key, value)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                #MODEL REGISTRY DOES NOT WORK with file store
                if tracking_url_type_store != "file":
                    
                    #RIGISTER THE MODEL
                    #there are other ways to use the model rigistry, which depends on the type of store
                    #please refer to the mlflow documentation for more details
                    #https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model" , registered_model_name = actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted) 
            
            logging.info(f"R2 score of the best model: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

    # def new_method(self, y_test, predicted_qualities):
    #     (rmse , mae , r2) = self.eval_metrics(y_test, predicted_qualities)
    #     return rmse,mae,r2