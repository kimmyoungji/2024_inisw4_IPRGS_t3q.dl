import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_DATA_PATH

import logging
import easyocr
import re
import torch
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import io
import base64
import inference_utils.license_plate as lp
from PIL import Image, ImageDraw

# 모델 클래스
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm",
                                                            num_labels=len({0: 'license_plate'}),
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

# 모델 초기화
def exec_init_model():
    logging.info('[hunmin log] the start line of the function [lp.exec_init_model]')
    logging.info('lp_model ocr reader ready')
    reader = easyocr.Reader(['ko'], gpu=True)

    logging.info('lp_model processor ready')
    processor_path = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(processor_path)

    logging.info('lp_model ready')
    license_model_path = f"{T3QAI_TRAIN_DATA_PATH}/models/lp_model.pt"
    license_model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    license_model.load_state_dict(torch.load(license_model_path))
    license_model.eval()

    model_info_dict = {
        "model": license_model,
        "processor": processor,
        "reader": reader
    }

    logging.info('[hunmin log] the end line of the function [lp.exec_init_model]')
    return model_info_dict

# 모델 추론 - 데이터프레임
def exec_inference_dataframe(df, seg_image, car_bbox, model_info_dict):
    
    logging.info('[hunmin log] the start line of the function [lp.exec_inference_dataframe]')    
    
    # 모델 준비
    lp_model = model_info_dict['licence_model']
    reader = model_info_dict['reader']

    # 프로세서 준비
    processor = model_info_dict['processor']

    # 입력데이터 준비
    img_base64 = df.iloc[0, 0]
    image_bytes = io.BytesIO(base64.b64decode(img_base64))
    image = Image.open(image_bytes).convert("RGBA")
    cropped_car = image.crop(car_bbox)
    cropped_car.save("/tmp/cropped_car.png")
    input = processor(cropped_car)

    # 모델 추론 요청
    output = lp_model(input)

    seg_lp_image, license_number, lp_error = lp.process(output,cropped_car,seg_image,car_bbox,reader)

    logging.info('[hunmin log] the end line of the function [lp.exec_inference_dataframe]')    
    return seg_lp_image, license_number, lp_error



# 모델 추론 - 파일
def exec_inference_file(files, model_info_dict):
    logging.info('[hunmin log] the start line of the function [exec_inference_file]')

    license_numbers = []

    for one_file in files:
        logging.info('[hunmin log] file inference')

        image = Image.open(one_file)
        encoding = model_info_dict['processor'](images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        pixel_mask = encoding["pixel_mask"]

        with torch.no_grad():
            output = model_info_dict['license_model'](pixel_values=pixel_values, pixel_mask=pixel_mask)

        original_size = torch.tensor([image.size[1], image.size[0]])

        # 예측된 결과에서 bounding box와 score를 추출하는 부분 추가
        processed_output = model_info_dict['processor'].post_process_object_detection(output, target_sizes=[original_size])[0]

        boxes = processed_output["boxes"].cpu().numpy()
        scores = processed_output["scores"].cpu().numpy()
        labels = processed_output["labels"].cpu().numpy()

        logging.info('lp_model ouput log...')
        logging.info({"boxes": boxes,"scores": scores,"labels": labels, })
        print("Boxes:", boxes)
        print("Scores:", scores)
        print("Labels:", labels)

        license_number = ""
        if len(boxes) == 0:
            license_numbers.append(license_number)
        else:
            # 여기에 license plate 인식 코드 추가
            xmin = boxes[0]
            ymin = boxes[1]
            xmax = xmin + boxes[2]
            ymax = ymin + boxes[3]

            cropped_plate = image.crop((xmin, ymin, xmax, ymax))
            # plt.imshow(cropped_plate)
            # plt.axis('off')
            # plt.show()
            result = model_info_dict['reader'].readtext(np.asarray(cropped_plate))
            
            for (_, text, _) in result:
                text = re.sub(r'[^가-힣0-9]', '', text)
                license_number += text
                print(license_number)
            license_numbers.append(license_number)
                
    return license_numbers

