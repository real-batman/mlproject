import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    AdaBoostClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_models, save_object

"This is the model trainer component which will train the model and save it in artifacts folder"

@dataclass
class ModelTrainerConfig:
    "This is the dataclass for ModelTrainerConfig which will contain the path of the trained model"
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:

    logging.info("Entered the Model Trainer component")
    "This class is responsible for training the model and saving it in artifacts folder"

    def __init__(self):
        "This is the constructor of the ModelTrainer class which initializes the ModelTrainerConfig dataclass"
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_arr, test_arr):
        "This is the method which will train the model and save it in artifacts folder"
        try:
            logging.info("Splitting training and testing input data")
            # Here we are splitting the training and testing data into input and target features
            # train_arr and test_arr are numpy arrays which are returned by the initiate_data_transformation method of the DataTransformation class

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # Here we are defining the models which we want to train and evaluate on the training and testing data
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "Adaboost Classifier": AdaBoostRegressor()
            }

            # Here we are evaluating the models on the training and testing data and storing the report in a dictionary
            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train,X_test = X_test, y_test = y_test, models = models)

            # Here we are getting the best model score from the report dictionary
            best_model_score = max(sorted(model_report.values()))

            # Here we are getting the best model name from the report dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Here we are getting the best model from the models dictionary
            best_model = models[best_model_name]

            logging.info(f"Best model found on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")

            if best_model_score < 0.6:
                logging.info("No best model found with score greater than 0.6")
                raise CustomException("No best model found with score greater than 0.6", sys)
            
            logging.info("Best found model on both training and testing dataset")

            # Here we are saving the best model in the artifacts folder using the save_object method of the utils.py file
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            logging.info("Error occurred in Model Trainer component")
            raise CustomException(e, sys)

    
#---- IGNORE THIS ----

'''
import os
import sys
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

def get_model_trainer_config():
    """Returns the configuration settings for the model trainer."""
    return {
        "trained_model_file_path": os.path.join('artifacts', 'model.pkl')
    }

def initiate_model_trainer(train_arr, test_arr):
    """
    Splits data, evaluates multiple models, and saves the best performing one.
    """
    try:
        logging.info("Splitting training and testing input data")
        
        # Slicing the numpy arrays: [rows, columns]
        # [:, :-1] takes all rows and all columns EXCEPT the last one (Features)
        # [:, -1] takes all rows and ONLY the last column (Target/Label)
        X_train, y_train, X_test, y_test = (
            train_arr[:, :-1],
            train_arr[:, -1],
            test_arr[:, :-1],
            test_arr[:, -1]
        )

        # Defining the "Tournament" contestants
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "Adaboost Regressor": AdaBoostRegressor()
        }

        logging.info("Evaluating models...")
        # evaluate_models returns a dict of {model_name: r2_score}
        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
        )

        # Finding the highest R2 score
        best_model_score = max(model_report.values())

        # Finding the name of the model that achieved that score
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        # Safety Threshold: If the best model is poor, we stop.
        if best_model_score < 0.6:
            logging.warning(f"Best model score {best_model_score} is below the 0.6 threshold")
            raise CustomException("No best model found with an acceptable score.")

        logging.info(f"Winner: {best_model_name} with R2 Score: {best_model_score}")

        # Saving the winning model brain
        config = get_model_trainer_config()
        save_object(
            file_path=config["trained_model_file_path"],
            obj=best_model
        )

        # Final validation check
        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)
        
        return r2_square

    except Exception as e:
        logging.error("Error occurred in Model Trainer function")
        raise CustomException(e, sys)

'''