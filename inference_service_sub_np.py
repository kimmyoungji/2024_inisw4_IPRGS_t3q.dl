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


license_numbers = []

def exec_inference_file(files):
    logging.info('[hunmin log] the start line of the function [exec_inference_file]')

    reader = easyocr.Reader(['ko'], gpu=True)
    print("OCR ready")

    processor_path = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(processor_path)

    license_model_path = f"{T3QAI_TRAIN_DATA_PATH}/model.pt"
    license_model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    license_model.load_state_dict(torch.load(license_model_path))
    license_model.eval()

    for one_file in files:
        logging.info('[hunmin log] file inference')

        inference_file = one_file
        print(inference_file)

        image = Image.open(one_file)
        encoding = processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        pixel_mask = encoding["pixel_mask"]

        with torch.no_grad():
            output = license_model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        original_size = torch.tensor([image.size[1], image.size[0]])

        # 예측된 결과에서 bounding box와 score를 추출하는 부분 추가
        processed_output = processor.post_process_object_detection(output, target_sizes=[original_size])[0]

        boxes = processed_output["boxes"].cpu().numpy()
        scores = processed_output["scores"].cpu().numpy()
        labels = processed_output["labels"].cpu().numpy()

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
          result = reader.readtext(np.asarray(cropped_plate))
          
          for (_, text, _) in result:
              text = re.sub(r'[^가-힣0-9]', '', text)
              license_number += text
              print(license_number)
          license_numbers.append(license_number)
              
    return license_numbers
