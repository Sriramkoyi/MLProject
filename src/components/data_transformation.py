import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()  

    def get_data_transformer(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            category_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                    ("scaler",StandardScaler())

                ]
            )
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("category_pipeline",category_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
                  raise Exception(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
         try:
              train_df=pd.read_csv(train_path)
              test_df=pd.read_csv(test_path)

              logging.info("Read train and test completed")
              logging.info("Obtaining preprocessing object")
              preprocessing_obj=self.get_data_transformer()
              target_column_name="math_score"
              
              input_train_df=train_df.drop(columns=[target_column_name],axis=1)
              target_train_df=train_df[target_column_name]

              input_test_df=test_df.drop(columns=[target_column_name],axis=1)
              target_test_df=test_df[target_column_name]

              logging.info("Applying preprocessing")
              input_train=preprocessing_obj.fit_transform(input_train_df)
              input_test=preprocessing_obj.transform(input_test_df)

              train_arr=np.c_[input_train,np.array(target_train_df)]
              test_arr=np.c_[input_test,np.array(target_test_df)]

              logging.info("Saved preprocessing object")

              save_object(
                   file_path=self.data_transformation_config.preprocessor_obj_file_path,
                   obj=preprocessing_obj
              )

              return [
                   train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
              ]
         except Exception as e:
              raise CustomException(e,sys)
              

