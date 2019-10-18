import os
import numpy as np
import torch
import torch.utils.data
import json
import cv2 as cv
import random
import transforms as T

class Positive_Roi_Dataset(torch.utils.data.Dataset):
    def __init__(self, root,train, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        imgs_list = list(sorted(os.listdir(os.path.join(root,'ROI_image'))))
        labels_list = list(sorted(os.listdir(os.path.join(root,'corres_labels_0to9'))))

        #全部的1212个文件作为索引值排序
        indices = [i for i in range(len(imgs_list))]
        #随机打乱顺序
        random.shuffle(indices)

        if train:
            self.imgs = [imgs_list[i] for i in indices[:-212]]
            self.labels = [labels_list[i] for i in indices[:-212]]
            if transforms == None:  #随机翻转图片
                transforms = T.Compose([T.ToTensor(),
                                        T.RandomHorizontalFlip(0.5)])
        else:
            self.imgs = [imgs_list[i] for i in indices[-212:]]
            self.masks = [labels_list[i] for i in indices[-212:]]
            if transforms == None:
                transforms = T.Compose([T.ToTensor()])
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images ad labels
        img_path = os.path.join(self.root, 'ROI_image', self.imgs[idx])
        labels_path = os.path.join(self.root, 'corres_labels_0to9', self.labels[idx])
        img = cv.imread(img_path)#[...,::-1]
        img = img.copy()
        #打开json文件，读取label坐标信息
        label_file = open(labels_path).read()
        label_list = json.loads(label_file)

        boxes = []
        #按照coco格式，写出标记的左上点和右下点
        #注意，坐标要计算相对距离，而不是全图坐标
        for i in range(1,len(label_list)):
            xmin = label_list[i]['x']-label_list[0]['x']
            ymin = label_list[i]['y']-label_list[0]['y']
            xmax = xmin + label_list[i]['w']
            ymax = ymin + label_list[i]['h']
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        #创建一个全为1（有无病变的二分类）的比当前json的列表少1（因为第一个是ROI而不是POS）的一维数组
        labels = torch.ones((len(label_list)-1), dtype=torch.int64)


        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(label_list)-1), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # target['img_name'] = img_path
        # target['labels_name'] =labels_path

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# dataset = Positive_Roi_Dataset('E:/ali_cervical_carcinoma_data',train=True)
#查看数据格式是否正确以及数据是否能对应上
# print(dataset[0][0].shape)#返回img,target，只查看target
# print(dataset[1][1])
# print(dataset[2][1])
# print(dataset[3][1])
# print(dataset[4][1])