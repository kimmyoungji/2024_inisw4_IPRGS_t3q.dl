import io
import base64
import logging
import inference_utils.inference_service_sub_od as od_inference
import inference_utils.inference_service_sub_seg as seg_inference
import inference_utils.inference_service_sub_lp as lp_inference
import postprocess_utils.area as area
from PIL import Image


def exec_init_model():
    logging.info('[hunmin log] the start line of the function [exec_init_model]')

    # Object Detection Model을 로드하고 초기화하는 메서드
    od_params = od_inference.exec_init_model()
    # Segmentation Model을 로드하고 초기화하는 메서드
    seg_params = seg_inference.exec_init_model()
    # License Plate Model을 로드하고 초기화하는 메서드
    lp_params = lp_inference.exec_init_model()
    
    logging.info('[hunmin log] the end line of the function [exec_init_model]')
    return {"od_params":{ **od_params },"seg_params":{ **seg_params },"lp_params":{ **lp_params }}


def exec_inference_dataframe(df, models_info_dict):
    logging.info('[hunmin log] the start line of the function [exec_inference_dataframe]')

    base64_string = df.iloc[0,0]
    decoded_image = base64.b64decode(base64_string)
    original_image = Image.open(io.BytesIO(decoded_image))

    # SEG MODEL
    try:
        # seg_inference.exec_inference_dataframe(df, model_info_dict) : Segmenation Model에 추론을 요청하는 메서드
        seg_image, areas, car_bbox, seg_result, seg_error = seg_inference.exec_inference_dataframe(df, models_info_dict['seg_params'])
        # logging.info(f'seg_model inference result: { {"seg_image": type(seg_image), "areas": areas, "car_bbox":car_bbox, "seg_result": seg_result, "seg_error":seg_error}}')
        logging.info(f'end of seg_model inference')

        if seg_error:
            return original_image, None, None, None, None, seg_error
        
    except Exception as e:
        return original_image, None, None, None, None, str(e) + " (in SEG)"

    # AREA MODEL
    try:
        # area.process : Segmentation Model 추론결과 얻어진 객체 마스크의 면적을 계산하는 메서드
        area_output = area.process(areas)
        # logging.info(f'area process result: { {"area_ouput":area_output}}')
        logging.info(f'end of area process')

    except Exception as e:
        return seg_image, seg_result, None, None, None, str(e) + " (in AREA)"

    # LP MODEL
    try:
        # lp_inference.exec_inference_dataframe(df...) : License Plate Model에 추론을 요청하는 메서드
        seg_lp_image, license_number, lp_error = lp_inference.exec_inference_dataframe(df, car_bbox, seg_image, models_info_dict['lp_params'])
        # logging.info(f'LP MODEL process result: { {"seg_lp_image":type(seg_lp_image), "license_number": license_number, "lp_error":lp_error}}')
        logging.info(f'end of lp_model inference')

        if lp_error:
            return seg_image, seg_result, area_output, None, None, lp_error  + " (in LP)"

    except Exception as e:
        return seg_image, seg_result, area_output, None, None, str(e)  + " (in LP)"

    # OD MODEL
    try:
        # od_inference.exec_inference_dataframe(df...) : Object Detection Model에 추론을 요청하는 메서드
        final_image, od_result, full_od_result, od_error = od_inference.exec_inference_dataframe(df,seg_image,models_info_dict['od_params'])
        # logging.info(f'OD MODEL process result: { {"final_image":type(final_image), "od_result": od_result, "full_od_result":full_od_result, "od_error":od_error}}')
        logging.info(f'end of od_model inference')

        if od_error:
            return seg_lp_image, seg_result, area_output, license_number, None, od_error + " (in OD)"
        
    except Exception as e:
        return seg_lp_image, seg_result, area_output, license_number, None, str(e) + " (in OD)"

    logging.info('[hunmin log] the end line of the function [exec_inference_dataframe]')
    # result
    return final_image, seg_result + od_result, area_output, license_number, full_od_result, None