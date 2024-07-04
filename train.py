import logging
import train
import t3qai_client as tc
import train_sub_lp as np
import train_sub_od as od


def main():
    result = None
    result_msg = "success"
    a = tc.t3qai_client()
    a.train_start()
    try:
        train()
    except Exception as e:
        result = e
        result_msg = e
        logging.info('error log : {}'.format(e))
    a.train_finish(result, result_msg)

def train():
    logging.info('[hunmin log] the start line of the function [train]')
    od.exec_train()
    np.exec_train()
    logging.info('[hunmin log] the end line of the function [train]')

if __name__ == '__main__':
    main()