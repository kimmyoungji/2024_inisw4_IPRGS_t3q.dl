# inference_service_sub.py
import os
import io
import base64
import logging
import numpy as np
from PIL import Image
import requests
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from inference_service_sub_seg2 import seg_process

# from t3qai_client import DownloadFile
# import t3qai_client as tc
# from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, \
#                             T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH

def exec_init_model():
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")

    model_info_dict = {
        "model": model,
        "processor": processor
    }
    return model_info_dict


def exec_inference_file(files, model_info_dict):
    
    """파일기반 추론함수는 files와 로드한 model을 전달받습니다."""
    logging.info('[hunmin log] the start line of the function [exec_inference_file]')
    
    # 모델, 인력값 프로세서 준비
    model = model_info_dict['model']
    processor = model_info_dict['processor']

    inference_result = []
    postprocessed_result = []
    
    # 모델 추론
    for one_file in files:
        logging.info(f'[hunmin log] inference: {one_file.filename}')
        
        # 모델 입력값 구성
        inference_file = one_file.file
        img = Image.open(inference_file)
        input = processor(images=img, return_tensors='pt')

        logging.info(f'[hunmin log] predict: {one_file.filename}')
        
        # 모델 추론 요청
        with torch.no_grad():
            output = model(**input)

            # 모델 추론 결과를 {segmentation, segment_info} 객체로 변환
            output2 = processor.post_process_panoptic_segmentation(output, target_sizes=[img.size[::-1]])[0]
            
            # 모델 추론 결과를 [{score, label, mask},...] 객체배열 로 변환
            output3 = []
            segmentations = output2['segmentation']
            segments_infos = output2['segments_info']
            for i in range(len(segments_infos)): 
                result_dict = {}
                result_dict['score'] = segments_infos[i]['score']
                segment_id = segments_infos[i]['label_id']
                result_dict['label'] = model.config.id2label[segment_id]
                result_dict['mask_image'] = get_mask(segment_id, segmentations)
                result_dict['mask'] = base64_encoding(get_mask(segment_id, segmentations))
                output3.append(result_dict)
            
            seg_image, areas, car_bbox, seg_result, seg_error = seg_process(img, output3)
            
    
    # 모델 추론 결과반환
    inference_result = output2
    postprocessed_result.append({ "seg_image": seg_image, "areas":areas, "car_bbox":car_bbox, "seg_result":seg_result, "seg_error":seg_error,\
                             "output3": output3})
    result = {'inference' :  inference_result, 'postprocessed': postprocessed_result }
    return result

# segmentations 텐서로부터, 각 세그먼트의 마스크를 생성
def get_mask(segment_id, segmentations):
  mask = (segmentations.numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask


import base64
from PIL import Image
from io import BytesIO

# 마스크를 base64형태로 변환
def base64_encoding(image):
    # Save the image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can choose the format you need, e.g., PNG, JPEG
    image_data = buffered.getvalue()

    # Encode the image data in base64
    base64_encoded = base64.b64encode(image_data).decode('utf-8')

    return base64_encoded