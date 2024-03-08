import pathlib
import os
import prediction_model


#__file__ points the __init__.py file of the package
PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE ='train.csv'
TEST_FILE ='test.csv'

MODEL_NAME='classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET_COLUMN ='Loan_Status'

# taken from jupyter notebook
FEATURES= ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area','CoapplicantIncome']

NUM_FEATURES=['ApplicantIncome','LoanAmount','Loan_Amount_Term']

CAT_FEATURES =['Gender', 
               'Married', 
               'Dependents',
               'Education',
               'Self_Employed',
               'Credit_History',
               'Property_Area']


# in our casr it is same as Categorical
FEATURES_TO_ENCODE =['Gender', 
               'Married', 
               'Dependents',
               'Education',
               'Self_Employed',
               'Credit_History',
               'Property_Area']

FEATURE_TO_MODIFY=['ApplicantIncome']

FEATURE_TO_ADD ='CoapplicantIncome'

DROP_FEATURES=['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome','LoanAmount'] # taking log of numerical columns