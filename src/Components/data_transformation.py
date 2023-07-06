import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os 
from src.utils import save_object

@dataclass

class DataTransformationConfig:
    preprocessing_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            # Define which columns should be ordinal-encoded and which should be scaled (Ordinal encoding B'coz Categorical value is in ranking value)
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # define the custon Rankings for each ordinal Variable
            cut_categories=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            clarity_categories=['I1', 'SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            color_categories=['D', 'E', 'F', 'G','H','I','J']
            
            logging.info("Pipeline Initiated")
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
                
                
            )

            ## categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                
            ])

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline, numerical_cols),
            ('cat_pipeline',cat_pipeline, categorical_cols)
                
            ])
            
            return preprocessor
            
            logging.info("Pipeline Completed")
            
            

        
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame Head : \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n{test_df.head().to_string()}")
            
            logging.info("Obtainging Preprocessing Object")
            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Transforming using Preprocessor Object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info("Applying preprocessing object on training and testing datasets")
            
            
            tarin_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj = preprocessing_obj
            )
            
            logging.info("Preprocessor pickel file saved")

            
        except Exception as e:
            logging.info("Exception Occured in the initiate_data_transformation ")
            
            raise CustomException(e,sys)
        

