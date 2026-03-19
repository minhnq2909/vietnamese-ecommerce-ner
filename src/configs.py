import os

# Data Paths
DATA_DIR = './data'
DATA_PATH_1 = os.path.join(DATA_DIR, 'data_label_1.csv')
DATA_PATH_2 = os.path.join(DATA_DIR, 'data_label_2.csv')
DATA_PATH_3 = os.path.join(DATA_DIR, 'data_label_3.csv')
PHOW2V_PATH = './data/word2vec_vi_words_300dims.txt' # Download from PhoW2V github repo

# Model Configs
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5

# Labels
LABEL_LIST = [
    'O', 'B-PROUCT_TYPE','I-PROUCT_TYPE',
    'B-PROUCT_NAME','I-PROUCT_NAME',
    'B-PRICE','I-PRICE',
    'B-LOCATION','I-LOCATION'
]
