import base64
import io
import logging, os
from ultralytics import YOLO
from t3qai_client import T3QAI_INIT_MODEL_PATH, T3QAI_TRAIN_MODEL_PATH
import t3qai_client as tc
from PIL import Image
import inference_utils.yolo_od as od

def exec_init_model():
    '''init_model에서 사용'''
    logging.info('[hunmin log] the start line of the function [od.exec_init_model]')

    model = YOLO(f"{tc.T3QAI_TRAIN_MODEL_PATH}/models/od_yolo_model.pt")
    model_info_dict = {
        "model": model
    }
    logging.info('[hunmin log] the end line of the function [od.exec_init_model]')
    return model_info_dict


def extract_labels(results, model_names):
    detected_objects = []
    for box in results[0].boxes:
        cls = int(box.cls[0].item())  # 클래스 번호 추출
        conf = box.conf[0].item()  # 신뢰도 값 추출
        detected_objects.append((model_names[cls], conf))
    return detected_objects


def exec_inference_dataframe(df, seg_image, model_info_dict):
    
    logging.info('[hunmin log] the start line of the function [od.exec_inference_dataframe]')

    try:
        ## 학습 모델 준비
        od_model = model_info_dict['model']
        
        # image preprocess
        img_base64 = df.iloc[0, 0]
        image_bytes = io.BytesIO(base64.b64decode(img_base64))
        image = Image.open(image_bytes)

        # 모델 추론 요청
        output = od_model(image)

        # final_image, od_result, full_od_result, None 반환
        final_image, od_result, full_od_result, od_error = od.process(seg_image, output, od_model)

        logging.info('[hunmin log] the end line of the function [od.exec_inference_dataframe]')
    except Exception as e:
        od_error = e  
    return final_image, od_result, full_od_result, od_error


def exec_inference_file(files, model_info_dict):
    """
    inference_file에서 사용
    파일기반 추론함수는 files와 로드한 model을 전달받습니다.
    """
    logging.info('[hunmin log] the start line of the function [exec_inference_file]')

    result = {}
    
    for idx, one_file in enumerate(files):
        # 파일 준비
        logging.info('[hunmin log] file inference')
        inference_file = one_file.file
        # 모델 가져오기 및 추론
        logging.info('[hunmin log] load model')
        output = model_info_dict['model'](inference_file)
        # 예외처리
        if len(output) == 0:
            return None, [], None, "Nothing detected in OD"
        # 결과 출력
        print(model_info_dict['model'].names)
        od_result = extract_labels(output, model_info_dict['model'].names)
        od_result = [label for label, _ in od_result]
        result[f'file{idx}_inference'] = od_result


    return result