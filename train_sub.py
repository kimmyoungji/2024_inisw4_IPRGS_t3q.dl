# Imports
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, \
                            T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH
import os
import logging
import meta_data.train_utils.train_sub_lp as lp
import meta_data.train_utils.train_sub_od as od
        
        
def exec_train():
    logging.info('[hunmin log] the start line of the function [exec_train]')
    logging.info('[hunmin log] T3QAI_TRAIN_DATA_PATH : {}'.format(T3QAI_TRAIN_DATA_PATH))
    od.exec_train()
    lp.exec_train()
    logging.info('[hunmin log] the end line of the function [exec_train]')
    
    
# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list)) 