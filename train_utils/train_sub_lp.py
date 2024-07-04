# Imports
import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_INIT_MODEL_PATH, T3QAI_MODULE_PATH, T3QAI_TRAIN_OUTPUT_PATH
import torch
import torchvision
import os
import logging
import matplotlib.pyplot as plt
import psutil
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm
import numpy as np

# 사용할 gpu 번호를 적는다.
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# 환경 변수 설정 (OpenMP 문제 해결)
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

processor_path = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(processor_path)

train_dataset = CocoDetection(img_folder=os.path.join(T3QAI_TRAIN_DATA_PATH,'dataset/lp/train'), processor=processor)
val_dataset = CocoDetection(img_folder=os.path.join(T3QAI_TRAIN_DATA_PATH,'dataset/lp/val'), processor=processor, train=False)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

def exec_train():
    logging.info('[hunmin log] the start line of the function [lp.exec_train]')
    logging.info('[hunmin log] T3QAI_TRAIN_DATA_PATH : {}'.format(T3QAI_TRAIN_DATA_PATH))
    
    # 저장 파일 확인
    list_files_directories(os.path.join(T3QAI_TRAIN_DATA_PATH,'dataset/lp'))
    
    batch = next(iter(train_dataloader))

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    # 학습 시작
    trainer = Trainer(max_epochs=1, gradient_clip_val=0.1)
    trainer.fit(model)

    ### 학습 모델의 성능을 검증하고 배포할 학습 모델을 저장
    ### 전처리 객체와 학습 모델 객체를 T3QAI_TRAIN_MODEL_PATH 에 저장
    model_save_path = os.path.join(T3QAI_TRAIN_MODEL_PATH, 'lp_model.pt')
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")
    
    # ### 학습 결과를 파일(이미지, 텍스트 등) 형태로 T3QAI_TRAIN_OUTPUT_PATH 에 저장 
    
    # 모델 로드
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()


    # 성능 평가 수행
    # Processor 로드
    processor_path = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(processor_path)

    # COCO 평가자 초기화
    evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

    for idx, batch in enumerate(tqdm(val_dataloader)):
        # 입력 데이터 가져오기
        pixel_values = batch["pixel_values"].to(model.device)
        pixel_mask = batch["pixel_mask"].to(model.device)
        labels = [{k: v.to(model.device) for k, v in t.items()} for t in batch["labels"]]  # DETR 형식, 크기 조정 및 정규화됨

        # 모델 예측
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # 예측 결과를 후처리
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

        # 평가자에 예측 결과 제공
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)

    # 평가 결과 요약
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    logging.info('[hunmin log] the end line of the function [lp.exec_train]')


# 모델 클래스
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm",
                                                            num_labels=len(id2label),
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list)) 


# 평가 함수 정의
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results
