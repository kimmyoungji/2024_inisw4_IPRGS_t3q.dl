import base64
import io
from PIL import Image
from io import BytesIO

import pandas as pd
from inference_service_sub import exec_inference_dataframe
import inference_service_sub_od as od_inference
import inference_service_sub_seg as seg_inference
import inference_service_sub_lp as lp_inference

import logging
logger = logging.getLogger()
logger.setLevel('INFO')

# 모델 초기화
def init_model():
    logging.info('[hunmin log] the start line of the function [init_model]')
    od_params = od_inference.exec_init_model()
    seg_params = seg_inference.exec_init_model()
    lp_params = lp_inference.exec_init_model()
    logging.info('[hunmin log] the end line of the function [init_model]')
    return {"od_params":{ **od_params },"seg_params":{ **seg_params },"lp_params":{ **lp_params }}

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

# 모델 추론 - file
# def inference_file(files, models_info_dict):
#     od_result = od_inference.exec_inference_file(files, models_info_dict['od_params'])
#     lp_result = lp_inference.exec_inference_file(files, models_info_dict['lp_params'])
#     seg_result = seg_inference.exec_inference_file(files, models_info_dict['seg_params'])
#     result = {'od_result':od_result,'seg_result':seg_result,'lp_result':lp_result}
#     logging.info('[hunmin log] the end line of the function [inference_file]')
#     return result

# 추론 실행
def main():

    # 모델 초기화
    models_info_dict = init_model()

    # 테스트 데이터셋 준비
    with open('./meta_data/dataset/test/00045.jpeg', 'rb') as file:
        data = file.read()
    encoded_data = base64.b64encode(data)
    data = [[encoded_data]]
    df = pd.DataFrame(data)

    # 모델 추론 요청
    result = inference_dataframe(df, models_info_dict)

    with open('./temp_result.txt', '+w') as file:
        file.write(str(result))

    # Convert bytes data to an image
    base64_string = result['image']
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))

    # Save the image as a PNG file
    image.save('final_image.png')

    logging.info('모델 추론 결과')
    logging.info(f'msg: {result["msg"]}')
    logging.info(f'od_result: {result["od_result"]}')
    logging.info(f'area: {result["area"]}')
    logging.info(f'license_number: {result["license_number"]}')

if __name__ == '__main__':
    main()