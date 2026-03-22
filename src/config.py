import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Bondora_preprocessed_outliers_replaced.csv')

# Features
NUM_COLS = ['Age', 'AppliedAmount', 'Interest', 'IncomeTotal', 'LiabilitiesTotal']
CAT_COLS = ['Education', 'EmploymentStatus']
USER_COL = 'UserName'

# Hyperparameters
MAX_SEQ_LEN = 6
BATCH_SIZE = 32
EPOCHS = 50
