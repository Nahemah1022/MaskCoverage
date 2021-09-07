import argparse
import os

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchvision
from torchvision import transforms, datasets, models

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import numpy as np
import matplotlib.patches as patches
import pandas as pd 

from tqdm import tqdm

from dset.mask_dataset import MaskDataset

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_model.py',
        description='Train object-detection model.',
    )
    parser.add_argument(
        '--batch_size',
        help='Evaluation batch size.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--epoch',
        help='Number of training epochs.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--lr',
        help='Gradient decent learning rate.',
        required=True,
        type=float,
    )
    parser.add_argument(
        '--max_norm',
        help='Gradient bound to avoid gradient explosion.',
        required=True,
        type=float,
    )
    parser.add_argument(
        '--workers',
        help='Number of dataloader worker.',
        required=True,
        type=int,
    )
    return parser.parse_args()
        
def collate_fn(batch):
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    args = parse_arg()


    imgs_path = r'data/images'
    labels_path = r'data/annotations/'

    data_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

    dataset = MaskDataset(imgs_path, labels_path, data_transform)

    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn,
                        num_workers=args.workers,
                  )

    model = get_model_instance_segmentation(3)
    model.train()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
                    params, 
                    lr=args.lr,
                    momentum=0.9, 
                    weight_decay=0.0005,
                )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    len_dataloader = len(data_loader)

    step = 0
    epoch_loss = 0
    pre_epochloss = 0

    for epoch in range(args.epoch):
        tqdm_dldr = tqdm(
            data_loader,
            desc=f'epoch: {epoch}, loss: {pre_epochloss:.6f}'
        )
        for i, (imgs, annotations) in enumerate(tqdm_dldr):
            
            optimizer.zero_grad()

            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            loss_dict = model([imgs[0]], [annotations[0]])
            losses = sum(loss for loss in loss_dict.values())   

            losses.backward()

            nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=args.max_norm,
            )

            optimizer.step()

            epoch_loss += losses

            step += 1

        tqdm_dldr.set_description(
            f'epoch: {epoch}, loss: {epoch_loss/i:.6f}'
        )

        writer.add_scalar('loss', epoch_loss/i, step)

        pre_epochloss = epoch_loss/i
        epoch_loss = 0

    # Save last checkpoint.
    torch.save(
        model.state_dict(),
        f'model.pt',
    )


if __name__ == '__main__':
    main()