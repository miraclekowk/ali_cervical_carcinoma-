import kfbReader
import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

''':parameter
1 kfbPath: kfb文件路径
2 scale: 缩放倍数
3 readAll:是否读取全部信息
'''

def ReadInfo(self,kfbPath,scale=0,readAll=False):
    return kfbReader._kfbReader.reader_ReadInfo(self,kfbPath,scale,readAll)

'''读取kfb文件信息demo'''
#参数设置
# path = r'E:/ali_cervical_carcinoma_data/neg_0/T2019_13.kfb'
# scale = 20
#
# #实例化reader类
# read = kfbReader.reader()
# kfb_information = kfbReader.reader.ReadInfo(read,path,scale,True)


#读取kfb文件的Roi
''':parameter
1 x: ROI起始位置横坐标
2 y: ROI起始位置纵坐标
3 w: ROI的Width信息
4 h: ROI的height信息
5 scale: 缩放倍数
:return
    numpy中的array类型'''

def ReadRoi(self,x,y,w,h,scale):
    arr = kfbReader._kfbReader.reader_ReadRoi(self,
                                              int(x),int(y),int(w),int(h),
                                              scale)
    arr = np.reshape(arr,[int(h),int(w),3].astype(np.uint8))
    return arr

#setReadScale 和getReadScale 的使用方法
def getReadScale(self):
    return kfbReader._kfbReader.reader_getReadScale(self)
def setReadScale(self,scale):
    return  kfbReader._kfbReader.reader_setReadScale(self.scale)


# #读取Roi位置
# # Roiroot= r'E:/ali_cervical_carcinoma_data/labels'
# roi = read.ReadRoi(10240,10240,512,512,scale)
#
# #显示Roi
# cv.imshow('roi',roi)
# cv.waitKey(0)  #不断刷新图像，延迟为Xms 等于0则只显示第一帧

#获取读到的kfb文件的相关信息
def getWidth(self):
    return  kfbReader._kfbReader.reader_getWidth(self)
def getHeight(self):
    return kfbReader._kfbReader.reader_getHeight(self)
def getReadScale(self):
    return kfbReader._kfbReader.reader_getReadScale(self)

# path = r'E:/ali_cervical_carcinoma_data/neg_0/T2019_13.kfb'
# Scale = 20
# #实例化reader类
# read = kfbReader.reader()
# kfbReader.reader.ReadInfo(read,path,Scale,True)
# width = kfbReader.reader.getWidth(read)
# height = kfbReader.reader.getHeight(read)
# print(width,height)
# #设置读取scale
# kfbReader.reader.setReadScale(read,scale=5)
# #读取ROI
# #roi_0~2  scale和坐标位置、大小等比例缩放
# roi_0 = read.ReadRoi(5120,5120,256,256,scale=5)
# roi_1 = read.ReadRoi(5120*2,5120*2,256*2,256*2,scale=10)
# roi_2 = read.ReadRoi(5120*4,5120*4,256*4,256*4,scale=20)
# # cv.imshow('',roi_0)
# # cv.waitKey(0)
# # cv.imshow('',roi_1)
# # cv.waitKey(0)
# # cv.imshow('',roi_2)
# # cv.waitKey(0)


# #roi_3~5  坐标位置、大小固定，scale改变
# roi_3 = read.ReadRoi(5120,5120,256,256,scale=5)
# roi_4 = read.ReadRoi(5120,5120,256,256,scale=10)
# roi_5 = read.ReadRoi(5120,5120,256,256,scale=20)

# for i in range(3,6):
#     plt.subplot(1,3,i-2)
#     Roi_image_str = str(f'roi_{i}')
#     print(Roi_image_str)
#     plt.imshow(locals()[Roi_image_str])
#     plt.title(Roi_image_str)
# plt.figure("roi_3~5  坐标位置、大小固定，scale改变")
# plt.show()



