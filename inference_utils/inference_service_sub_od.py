import base64
import io
import logging, os
from ultralytics import YOLO
from PIL import Image
import postprocess_utils.yolo_od as od

def exec_init_model(T3QAI_INIT_MODEL_PATH):
    '''init_model에서 사용'''
    logging.info('[hunmin log] the start line of the function [od.exec_init_model]')

    model = YOLO(os.path.join(T3QAI_INIT_MODEL_PATH,'od_yolo_model.pt'))
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