# preprocess.py
from preprocess_sub import exec_process
import logging


def process_for_train(pm):
    exec_process(pm)
    logging.info('[hunmin log] the end line of the function [process_for_train]')
    

def init_svc(im, rule):
    return {}


def transform(df, params, batch_id):
    logging.info('[hunmin log] df.shape : {}'.format(df.shape))
    logging.info('[hunmin log] type(df) : {}'.format(type(df)))
    logging.info('[hunmin log] the end line of the function [transform]')
    return df