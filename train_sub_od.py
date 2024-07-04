# Imports
import t3qai_client as tc
import torch
import os
import logging
import matplotlib.pyplot as plt
from ultralytics import YOLO
import psutil

# 사용할 gpu 번호를 적는다.
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# 환경 변수 설정 (OpenMP 문제 해결)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.info(f'[hunmin log] torch ver : {torch.__version__}')

# Check if CUDA-enabled GPUs are available
if torch.cuda.is_available():
    try:
        num_gpus = torch.cuda.device_count()
        logging.info('[hunmin log] GPU set complete')
        logging.info('[hunmin log] Number of GPUs: {}'.format(num_gpus))
    except RuntimeError as e:
        logging.info('[hunmin log] GPU set failed')
        logging.info(e)
else:
    logging.info('[hunmin log] No CUDA-enabled GPU available')


# exec_train
def exec_train():
    logging.info('[hunmin log] the start line of the function [od.exec_train]')
    logging.info('[hunmin log] T3QAI_TRAIN_DATA_PATH : {}'.format(tc.T3QAI_TRAIN_DATA_PATH))
    
    # 저장 파일 확인
    list_files_directories(tc.T3QAI_TRAIN_DATA_PATH)
    
    yaml_path = f"{tc.T3QAI_TRAIN_DATA_PATH}/dataset.yaml"

    dataset_yaml = f"""
        train: '{tc.T3QAI_TRAIN_DATA_PATH}/dataset/od/train/images'
        val: '{tc.T3QAI_TRAIN_DATA_PATH}/dataset/od/val/images'

        names: ['Road_No_Parking', 'Road_Speed_Limit_in_School_Zone', 'Road_School_Zone', 'Crosswalk',  'Road_No_Stopping_or_Parking',  'Road_No_Stopping_Zone', 'stop', 'traffic_lane_yellow_solid',  'school_zone',  'no_parking',  'fire_hydrant']

        train_annotation: '{tc.T3QAI_TRAIN_DATA_PATH}/dataset/od/train/labels'
        val_annotation: '{tc.T3QAI_TRAIN_DATA_PATH}/dataset/od/val/labels'
    """

    with open(yaml_path, 'w') as file:
        file.write(dataset_yaml)
    
    ### 데이터 전처리와 학습 모델을 구성하고 모델 학습을 수행
    model = YOLO(f'{tc.T3QAI_TRAIN_MODEL_PATH}/models/yolov8n.pt')

    # 시스템 메모리와 CPU 사용량 모니터링
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")

    #학습 시작
    model.train(
        data=yaml_path,
        epochs=1,
        imgsz=640,
        batch=32,       # 배치 크기 줄이기
        workers=6,      # 워커 수 줄이기
        name='yolov8_multiclass',
        cache=True,     # 캐시 사용
        project= os.path.join(tc.T3QAI_TRAIN_OUTPUT_PATH, 'logs'), # '.\\multi_V8\\logs',
        exist_ok=False
    )
    
    ## 학습 모델의 성능을 검증하고 배포할 학습 모델을 저장
    ## 전처리 객체와 학습 모델 객체를 T3QAI_TRAIN_MODEL_PATH 에 저장
    model.export(format="onnx")  # creates 'yolov8n.onnx'

    ## 학습 결과를 파일(이미지, 텍스트 등) 형태로 T3QAI_TRAIN_OUTPUT_PATH 에 저장 
    
    # 모델 로드
    logfiles = os.listdir(f'{tc.T3QAI_TRAIN_MODEL_PATH}/logs')
    for i in range(len(logfiles)):
        if logfiles[i] == 'yolov8_multiclass':
            logfiles[i] = 'yolov8_multiclass0'
    logfiles_list = [ (int(logfile.removeprefix('yolov8_multiclass')), logfile) for logfile in logfiles ]
    logfiles_list.sort()
    model = YOLO(f"{tc.T3QAI_TRAIN_MODEL_PATH}/logs/{logfiles_list[-1][1]}/weights/best.pt")

    # 성능 평가 수행
    results = model.val(data=yaml_path, conf=0.25, iou=0.5)
    logging.info('결과')
    logging.info(results.confusion_matrix)

    # 모델 저장
    model.save(f'{tc.T3QAI_MODULE_PATH}/models/od_yolo_model.pt')

    # 결과를 파일로 저장
    #os.makedirs(T3QAI_TRAIN_OUTPUT_PATH, exist_ok=True)
    with open(os.path.join(tc.T3QAI_TRAIN_OUTPUT_PATH, 'od_evaluation_results.txt'), 'w') as f:
        f.write("Evaluation Results:")
        f.write(f"Precision: {results.results_dict}\n")
    
    logging.info('[hunmin log] the end line of the function [od.exec_train]')


###########################################################################
## exec_train() 호출 함수 
###########################################################################
      
# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))

