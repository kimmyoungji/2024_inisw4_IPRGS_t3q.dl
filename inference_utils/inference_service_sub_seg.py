# inference_service_sub.py
import os
import io
import base64
import logging
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import postprocess_utils.segmentations as seg
from io import BytesIO

# from t3qai_client import DownloadFile
# import t3qai_client as tc
# from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, \
#                             T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH

def exec_init_model():
    logging.info('[hunmin log] the start line of the function [seg.exec_init_model]')
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")

    model_info_dict = {
        "model": model,
        "processor": processor
    }
    logging.info('[hunmin log] the end line of the function [seg.exec_init_model]')
    return model_info_dict


def exec_inference_dataframe(df, model_info_dict):
    
    logging.info('[hunmin log] the start line of the function [seg.exec_inference_dataframe]')    

    # 모델 준비
    model = model_info_dict['model']

    # 프로세서준비
    processor = model_info_dict['processor']

    # 이미지 준비
    try: 
        img_base64 = df.iloc[0, 0]
        decoded_img = base64.b64decode(img_base64)
        image_bytes = io.BytesIO(decoded_img)
        image = Image.open(image_bytes).convert('RGB')
        input = processor(images=image, return_tensors='pt')
    except Exception as e:
        logging.exception(e)

    # 모델 추론 요청
    with torch.no_grad():
        output = model(**input)

        # 모델 추론 결과를 {segmentation, segment_info} 객체로 변환
        output2 = processor.post_process_panoptic_segmentation(output, target_sizes=[image.size[::-1]])[0]
        
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
        
        seg_image, areas, car_bbox, seg_result, seg_error = seg.process(image, output3)

    logging.info('[hunmin log] the end line of the function [seg.exec_inference_dataframe]')
    return seg_image, areas, car_bbox, seg_result, seg_error


# segmentations 텐서로부터, 각 세그먼트의 마스크를 생성
def get_mask(segment_id, segmentations):
  mask = (segmentations.numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask


# 마스크를 base64형태로 변환
def base64_encoding(image):
    # Save the image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can choose the format you need, e.g., PNG, JPEG
    image_data = buffered.getvalue()

    # Encode the image data in base64
    base64_encoded = base64.b64encode(image_data).decode('utf-8')

    return base64_encoded