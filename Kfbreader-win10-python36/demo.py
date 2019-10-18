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

num_classes=2
# faster_rcnn.resnet_fpn_backbone内部将backbone的第1, 第2卷积层冻结，不参与更新
backbone = faster_rcnn.resnet_fpn_backbone(backbone_name='resnet50', pretrained=True)
rpn_anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                   aspect_ratios=((0.5, 1.0, 2.0),) * 5)
model = faster_rcnn.FasterRCNN(backbone=backbone, num_classes=num_classes, min_size=600, max_size=600, rpn_anchor_generator=rpn_anchor_generator)

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root='datasets/PennFudanPed', train=True, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        indices = [i for i in range(len(imgs))]
        random.shuffle(indices)
        if train:
            self.imgs = [imgs[i] for i in indices[:-50]]
            self.masks = [masks[i] for i in indices[:-50]]
            if transforms == None:
                transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
        else:
            self.imgs = [imgs[i] for i in indices[-50:]]
            self.masks = [masks[i] for i in indices[-50:]]
            if transforms == None:
                transforms = T.Compose([T.ToTensor()])          
        self.transforms = transforms


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


data_train = PennFudanDataset(train=True)
data_test = PennFudanDataset(train=False)
print('data_train num=', len(data_train), '\nfileds:\n', data_train[0][1])
# print('data_test num=', len(data_test), '\nfileds:\n', data_test[0][1])
trainLoader = data.DataLoader(data_train, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
testLoader = data.DataLoader(data_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)


device = torch.device('cuda')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

for epoch in range(10):
    train_one_epoch(model, optimizer, trainLoader, device, epoch, print_freq=10)
    lr_scheduler.step()
    
    utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),},
                os.path.join('checkpoints', 'model_{}.pth'.format(epoch))) 
                   
    evaluate(model, testLoader, device=device)
