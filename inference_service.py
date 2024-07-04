import base64
import io
from inference_service_sub import exec_init_model, exec_inference_dataframe

import logging
logger = logging.getLogger()
logger.setLevel('INFO')


# 모델 초기화
def init_model():
    logging.info('[hunmin log] the start line of the function [init_model]')

    models_info_dict = exec_init_model()
    
    logging.info('[hunmin log] the end line of the function [init_model]')
    return { **models_info_dict }


# 모델 추론 - dataframe
def inference_dataframe(df, models_info_dict):
    logging.info(f'[hunmin log] the start line of the function [inference_dataframe]')

    final_image, od_result, area_output, license_number, full_od_result, error = exec_inference_dataframe(df, models_info_dict)
    response_data = {
        'msg': error if error else "success",
        'image': pil_image_to_base64(final_image) if final_image else None,
        'od_result': od_result if od_result else [],
        'area': area_output if area_output else {},
        'license_number': license_number if license_number else ""
    }
    
    logging.info(f'[hunmin log] the end line of the function [inference_dataframe].')
    return response_data


def pil_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str