import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import numpy as np
import matplotlib.patches as patches
import pandas as pd 

import random
import cv2
import sys
sys.path.append('./')
import time

def parse_arg():
    parser = argparse.ArgumentParser(
        prog='python demo.py', 
        description='Pytorch Faster-rcnn Detection'
    )
    parser.add_argument(
        '--model_path', 
        help='model path',
        type=str, 
    )
    parser.add_argument(
        '--image_path', 
        help='image path',
        type=str, 
    )
    return parser.parse_args()
 
def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)

def get_model_instance_segmentation(num_classes):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

@torch.no_grad()
def main(image_url, show_img=True):
    if not image_url:
        image_url = args.image_path
    args = parse_arg()
    input = []
    num_classes = 91

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    origin_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    origin_model = origin_model.to(device)
    origin_model.eval()

    src_img = cv2.imread(image_url)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(device)

    input.append(img_tensor)
    out = origin_model(input)

    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    time.sleep(10)
    model = get_model_instance_segmentation(3)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)
    
    out2 = model(input)

    boxes2 = out2[0]['boxes']
    labels2 = out2[0]['labels']
    scores2 = out2[0]['scores']
    
    num_person = 0
    num_person_with_mask = 0
    for idx in range(boxes2.shape[0]):
        if scores2[idx] >= 0.8:
            x1, y1, x2, y2 = int(boxes2[idx][0]), int(boxes2[idx][1]), int(boxes2[idx][2]), int(boxes2[idx][3])
            cv2.rectangle(src_img,(x1,y1),(x2,y2),random_color(),thickness=2)
            num_person_with_mask = num_person_with_mask + 1

    for idx in range(boxes.shape[0]):
        if labels[idx] != 1:
                continue
        if scores[idx] >= 0.8:
            num_person = num_person + 1
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
            name = 'Person'
            probability = str(float(scores[idx])*100).split('.')
            probability = probability[0] + '.' + probability[1][0] + '%'
            color = random_color()
            cv2.rectangle(src_img,(x1,y1),(x2,y2),color,thickness=2)
            cv2.rectangle(src_img,(x1-1, y1-13),(x1+72,y1),color,thickness=cv2.FILLED)
            cv2.putText(src_img, text=name, org=(x1+2, y1-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.3, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))
            cv2.putText(src_img, text=probability, org=(x1+40, y1-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.3, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

    str1 = str(num_person_with_mask) + '/' + str(num_person)
    if num_person!= 0:
        cv2.rectangle(src_img,(0, 0),(65,20),(0,0,0),thickness=cv2.FILLED)
        cv2.putText(src_img, str1, org=(5, 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 0))
    
    else :
        cv2.rectangle(src_img,(0, 0),(80,20),(0,0,0),thickness=cv2.FILLED)
        cv2.putText(src_img, 'no person', org=(2, 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 0))
    
    if show_img:
        cv2.imshow('People with mask',src_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        predictpath = '../media/result/'+ image_url.split('/')[-1]
        cv2.imwrite(predictpath, src_img)
        return predictpath
 
if __name__ == "__main__":
    main()