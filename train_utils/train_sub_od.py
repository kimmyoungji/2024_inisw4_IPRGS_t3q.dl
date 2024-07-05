import os
import logging
from ultralytics import YOLO
import psutil


# Object Detection 모델의 학습을 시작하는 코드
def exec_train(T3QAI_TRAIN_DATA_PATH,T3QAI_TRAIN_OUTPUT_PATH,T3QAI_TRAIN_MODEL_PATH):
    logging.info('[hunmin log] the start line of the function [od.exec_train]')
    logging.info('[hunmin log] T3QAI_TRAIN_DATA_PATH : {}'.format(T3QAI_TRAIN_DATA_PATH))
    
    # 저장 파일 확인
    list_files_directories(os.path.join(T3QAI_TRAIN_DATA_PATH,'dataset/od'))
    
    yaml_relative_path = os.path.join(T3QAI_TRAIN_DATA_PATH,'dataset.yaml')
    yaml_absolute_path = os.path.abspath(yaml_relative_path)

    T3QAI_TRAIN_DATA_ABS_PATH = os.path.abspath(T3QAI_TRAIN_DATA_PATH)

    dataset_yaml = f"""
        train: '{os.path.join(T3QAI_TRAIN_DATA_ABS_PATH,'dataset/od/train/images')}'
        val: '{os.path.join(T3QAI_TRAIN_DATA_ABS_PATH,'dataset/od/val/images')}'

        names: ['Road_No_Parking', 'Road_Speed_Limit_in_School_Zone', 'Road_School_Zone', 'Crosswalk',  'Road_No_Stopping_or_Parking',  'Road_No_Stopping_Zone', 'stop', 'traffic_lane_yellow_solid',  'school_zone',  'no_parking',  'fire_hydrant']

        train_annotation: '{os.path.join(T3QAI_TRAIN_DATA_ABS_PATH,'dataset/od/train/labels')}'
        val_annotation: '{os.path.join(T3QAI_TRAIN_DATA_ABS_PATH,'dataset/od/val/labels')}'
    """

    with open(yaml_absolute_path, 'w') as file:
        file.write(dataset_yaml)
    
    ### 데이터 전처리와 학습 모델을 구성하고 모델 학습을 수행
    model = YOLO("yolov8n.pt")

    # 시스템 메모리와 CPU 사용량 모니터링
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")

    #학습 시작
    model.train(
        data=yaml_absolute_path,
        epochs=1,
        imgsz=640,
        batch=32,       
        workers=6,      
        name='yolov8_multiclass',
        cache=True,  
        project= os.path.join(T3QAI_TRAIN_OUTPUT_PATH, 'od_yolo_model_logs'), 
        exist_ok=False
    )
    
    ## 학습 모델의 성능을 검증하고 배포할 학습 모델을 저장
    model.export(format="onnx")  # creates 'yolov8n.onnx'
    
    # 모델 저장
    model_save_path = os.path.join(T3QAI_TRAIN_MODEL_PATH, 'od_yolo_model.pt')
    model.save(model_save_path)

    # 모델 로드
    model = YOLO(model_save_path)

    # 성능 평가 수행
    results = model.val(data=yaml_absolute_path, conf=0.25, iou=0.5)
    logging.info('Object Detection Model training result')
    logging.info(results.confusion_matrix)
    
    # 결과를 파일로 저장
    with open(os.path.join(T3QAI_TRAIN_OUTPUT_PATH, 'od_evaluation_results.txt'), 'w') as f:
        f.write("Evaluation Results:")
        f.write(f"Precision: {results.results_dict}\n")
    
    logging.info('[hunmin log] the end line of the function [od.exec_train]')


# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))

