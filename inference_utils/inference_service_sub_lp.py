import logging
import easyocr
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import pytorch_lightning as pl
import io
import base64
import postprocess_utils.license_plate as lp
from PIL import Image
import os

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
def exec_init_model(T3QAI_INIT_MODEL_PATH):
    logging.info('[hunmin log] the start line of the function [lp.exec_init_model]')

    reader = easyocr.Reader(['ko'], gpu=True)

    processor_path = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(processor_path)

    lp_model_path = os.path.join(T3QAI_INIT_MODEL_PATH, 'lp_model.pt')
    lp_model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    lp_model.load_state_dict(torch.load(lp_model_path))
    lp_model.eval()

    lp_model_info_dict = {
        "model": lp_model,
        "processor": processor,
        "reader": reader
    }

    logging.info('[hunmin log] the end line of the function [lp.exec_init_model]')
    return lp_model_info_dict


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
