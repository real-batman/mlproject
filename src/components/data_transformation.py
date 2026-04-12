import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    # This dataclass is used to define the configuration for data transformation, specifically the file path where the preprocessor object will be saved.
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    # The DataTransformation class is responsible for handling the data transformation process, 
    # which includes creating a preprocessor object and applying it to the training and testing datasets.

    def __init__(self):
        # __init_ method is used to initialize the DataTransformationConfig dataclass and assign it to an instance variable.
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformaer_object(self):
        # This method is responsible for creating and returning a preprocessor object that can be used to transform the data.
        try:
            #defining numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Creating numerical pipeline
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Creating categorical pipeline
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_emcoder", OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical Standard Scaling Completed')
            logging.info('Categorical Encoding Completed')

            # Combining numerical and categorical pipeline
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),   #Syntax: ('name of the pipeline', pipeline_object, list of columns to apply the pipeline on)
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            # returning the preprocessor object
            return preprocessor

        except Exception as e:
            logging.info("Error occurred in Data Transformation get_data_transformer_object")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        # This method is responsible for reading the training and testing data, applying the preprocessor object to transform the data, and saving the preprocessor object for future use.
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformaer_object()

            target_column_name = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =  preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            logging.info("Error occurred in Data Transformation initiate_data_transformation")
            raise CustomException(e, sys)
        




#-------- IGNORE THIS ---------
'''
import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

def get_data_transformation_config():
    """Returns the configuration for data transformation artifacts."""
    return {
        "preprocessor_obj_file_path": os.path.join('artifacts', "proprocessor.pkl")
    }

def get_data_transformer_object():
    """Creates and returns the preprocessor object."""
    try:
        # Define numerical and categorical columns
        numerical_columns = ["writing_score", "reading_score"]
        categorical_columns = [
            "gender", "race_ethnicity", "parental_level_of_education",
            "lunch", "test_preparation_course"
        ]

        # Pipeline for numerical data: Handle missing values -> Standardize
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Pipeline for categorical data: Handle missing values -> Encode -> Standardize
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])

        logging.info("Created numerical and categorical pipelines")

        # Combine pipelines into one preprocessor
        preprocessor = ColumnTransformer([
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipelines", cat_pipeline, categorical_columns)
        ])

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)

def initiate_data_transformation(train_path, test_path):
    """Reads data, applies transformation, and saves the preprocessor object."""
    try:
        config = get_data_transformation_config()
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Read train and test data completed")

        # Get the preprocessor tool
        preprocessing_obj = get_data_transformer_object()

        target_column_name = "math_score"

        # Separate Features (X) and Target (y)
        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]

        logging.info("Applying preprocessing object on training and testing dataframes")

        # Transform the features
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        # Recombine Features and Target into a single array for the model
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        # Save the preprocessor object for future use (Prediction pipeline)
        save_object(
            file_path=config["preprocessor_obj_file_path"],
            obj=preprocessing_obj
        )

        return (
            train_arr,
            test_arr,
            config["preprocessor_obj_file_path"]
        )

    except Exception as e:
        raise CustomException(e, sys)
'''