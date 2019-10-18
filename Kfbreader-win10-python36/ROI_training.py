import os
import random
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils import data
import transforms as T
import utils
from engine import evaluate, train_one_epoch
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from dataset_maker import Positive_Roi_Dataset
from gpu_mem_track import MemTracker
import inspect
num_classes = 2
# faster_rcnn.resnet_fpn_backbone内部将backbone的第1, 第2卷积层冻结，不参与更新
backbone = faster_rcnn.resnet_fpn_backbone(backbone_name='resnet50', pretrained=False)
rpn_anchor_generator = AnchorGenerator(sizes=((8,), (16,), (32,), (64,), (128,)),
                                       aspect_ratios=((0.5, 1.0, 2.0),) * 5)
model = faster_rcnn.FasterRCNN(backbone=backbone, num_classes=num_classes, min_size=600, max_size=600,
                               rpn_anchor_generator=rpn_anchor_generator)



data_train =Positive_Roi_Dataset('E:/ali_cervical_carcinoma_data',train=True)
data_test =Positive_Roi_Dataset('E:/ali_cervical_carcinoma_data',train=False)
# print('data_test num=', len(data_test), '\nfileds:\n', data_test[0][1])
trainLoader = data.DataLoader(data_train, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
testLoader = data.DataLoader(data_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
torch.cuda.empty_cache()

#观察GPU
device = torch.device('cuda')

frame = inspect.currentframe()  # define a frame to track
gpu_tracker = MemTracker(frame)  # define a GPU tracker

gpu_tracker.track()  # run function between the code line where uses GPU
model.to(device)
# gpu_tracker.track()  # run function between the code line where uses GPU


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.9, weight_decay=0.0005)


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

gpu_tracker.track()
for epoch in range(1, 10):

    train_one_epoch(model, optimizer, trainLoader, device, epoch, print_freq=10)
    torch.cuda.empty_cache()
    lr_scheduler.step()
    gpu_tracker.track()
    torch.cuda.empty_cache()
    utils.save_on_master({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(), },
        os.path.join('checkpoints', 'model_{}.pth'.format(epoch)))
    # torch.cuda.empty_cache()
    evaluate(model, testLoader, device=device)
    # torch.cuda.empty_cache()
    gpu_tracker.track()


