import logging, os
from ultralytics import YOLO
from t3qai_client import T3QAI_INIT_MODEL_PATH, T3QAI_TRAIN_MODEL_PATH

def exec_init_model():
  '''init_model에서 사용'''
  model_path = os.path.join(T3QAI_INIT_MODEL_PATH, 'yolov8_multiclass.pt')
  model = YOLO(model_path)
  # model_info_dict = {
  #     "model": model
  # }
  return model

def extract_labels(results, model_names):
      detected_objects = []
      for box in results[0].boxes:
          cls = int(box.cls[0].item())  # 클래스 번호 추출
          conf = box.conf[0].item()  # 신뢰도 값 추출
          detected_objects.append((model_names[cls], conf))
      return detected_objects

def exec_inference_file(files):
    """
    inference_file에서 사용
    파일기반 추론함수는 files와 로드한 model을 전달받습니다.
    """
    logging.info('[hunmin log] the start line of the function [exec_inference_file]')

    # model_path = os.path.join(T3QAI_INIT_MODEL_PATH, 'best.pt')
    
    model_path = f"{T3QAI_TRAIN_MODEL_PATH}/logs/yolov8_multiclass/weights/best.pt"
    model = YOLO(model_path)

    result = {}
    
    for idx, one_file in enumerate(files):
        # 파일 준비
        logging.info('[hunmin log] file inference')
        inference_file = one_file.file
        # 모델 가져오기 및 추론
        logging.info('[hunmin log] load model')
        output = model(inference_file)
        # 예외처리
        if len(output) == 0:
          return None, [], None, "Nothing detected in OD"
        # 결과 출력
        print(model.names)
        od_result = extract_labels(output, model.names)
        od_result = [label for label, _ in od_result]
        result[f'file{idx}_inference'] = od_result

#     result = [DownloadFile(file_path=T3QAI_TRAIN_OUTPUT_PATH+'/Accuracy_Loss.png', file_name='result.jpg'), 
#               DownloadFile(file_path=T3QAI_TRAIN_OUTPUT_PATH+'/Accuracy_Loss.png', file_name='result2.jpg')]

    return result

