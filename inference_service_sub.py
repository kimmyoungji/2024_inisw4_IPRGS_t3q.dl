import logging, os
from ultralytics import YOLO
from t3qai_client import T3QAI_INIT_MODEL_PATH, T3QAI_TRAIN_MODEL_PATH

import inference_service_sub_od as od_inference
import inference_service_sub_seg as seg_inference
import inference_service_sub_lp as lp_inference
import inference_utils.area as area


def exec_inference_dataframe(df, models_info_dict):

    # SEG MODEL
    try:
        seg_image, areas, car_bbox, seg_result, seg_error = seg_inference.exec_inference_dataframe(df, models_info_dict['seg_params'])

        logging.info(f'seg_model inference result: { {"seg_image": type(seg_image), "areas": areas, "car_bbox":car_bbox, "seg_result": seg_result, "seg_error":seg_error}}')

        if seg_error:
            return None, None, None, None, None, seg_error
        
    except Exception as e:
        return None, None, None, None, None, str(e) + " (in SEG)"

    # AREA MODEL
    try:
        area_output = area.process(areas)
        logging.info(f'area process result: { {"area_ouput":area_output}}')

    except Exception as e:
        return seg_image, seg_result, None, None, None, str(e) + " (in AREA)"

    # LP MODEL
    try:
        seg_lp_image, license_number, lp_error = lp_inference.exec_inference_dataframe(df, car_bbox, seg_image, models_info_dict['lp_params'])
        logging.info(f'LP MODEL process result: { {"seg_lp_image":type(seg_lp_image), "license_number": license_number, "lp_error":lp_error}}')

        if lp_error:
            return seg_image, seg_result, area_output, None, None, lp_error  + " (in LP)"

    except Exception as e:
        return seg_image, seg_result, area_output, None, None, str(e)  + " (in LP)"

    # OD MODEL
    try:
        final_image, od_result, full_od_result, od_error = od_inference.exec_inference_dataframe(df,seg_image,models_info_dict['od_params'])
        logging.info(f'OD MODEL process result: { {"final_image":type(final_image), "od_result": od_result, "full_od_result":full_od_result, "od_error":od_error}}')

        if od_error:
            return seg_lp_image, seg_result, area_output, license_number, None, od_error + " (in OD)"
        
    except Exception as e:
        return seg_lp_image, seg_result, area_output, license_number, None, str(e) + " (in OD)"

    # result
    return final_image, seg_result + od_result, area_output, license_number, full_od_result, None
    