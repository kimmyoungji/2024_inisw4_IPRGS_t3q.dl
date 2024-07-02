import inference_service_sub_od as od_inference
import inference_service_sub_seg as seg_inference
import inference_service_sub_np as np_inference

import logging
logger = logging.getLogger()
logger.setLevel('INFO')

def init_model():
    od_params = od_inference.exec_inference_file(files, model_info_dict)
    seg_params = seg_inference.exec_init_model()
    np_params = np_inference.exec_init_model()
    logging.info('[hunmin log] the end line of the function [init_model]')
    return {"od_params":{ **od_params },"seg_params":{ **seg_params },"np_params":{ **np_params }}

# def inference_dataframe(df, model_info_dict):
#     result = exec_inference_dataframe(df, model_info_dict)
#     logging.info('[hunmin log] the end line of the function [inference_dataframe]')
#     return { **result }

def inference_file(files, model_info_dict):
    od_result = od_inference.exec_inference_file(files, model_info_dict)
    seg_result = seg_inference.exec_inference_file(files, model_info_dict)
    np_result = np_inference.exec_inference_file(files, model_info_dict)
    result = {'od_result':od_result,'seg_result':seg_result,'np_result':np_result}
    logging.info('[hunmin log] the end line of the function [inference_file]')
    return result