import logging
from train_sub import exec_train
from t3qai_client import tc


def main():
    result = None
    result_msg = "success"
    tc.train_start()
    try:
        train()
    except Exception as e:
        result = e
        result_msg = e
        logging.info('error log : {}'.format(e))
    tc.train_finish(result, result_msg)


def train():
    try:
        exec_train()
    except Exception as e:
        logging.info('error log : {}'.format(e))
    logging.info('[hunmin log] the end line of the function [train]')


if __name__ == '__main__':
    main()