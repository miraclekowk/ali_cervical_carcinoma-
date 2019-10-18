import kfbReader
import json
import os
import cv2 as cv
import time

total_time  = 0

#计算相对坐标
def caculate_relative_position(Roi_x,Roi_y,
                               Pos_x,Pos_y,):
    relative_x = Pos_x - Roi_x
    relative_y = Pos_y - Roi_y
    return relative_x,relative_y


# 选中Roi且画框，向函数传递正在处理的labels文件名和相应的json的List
def draw_rectangle(labels_filename, corres_json_list, total_time):
    start_time = time.time()  # 完成画一张图记一次时间
    #读取图像
    filename = labels_filename[:-10] + '.kfb'
    Roi_x = corres_json_list[0]['x']
    Roi_y = corres_json_list[0]['y']
    Roi_w = corres_json_list[0]['w']
    Roi_h = corres_json_list[0]['h']
    # 实例化reader类
    path = os.path.join(kfb_image_root, filename)
    image = kfbReader.reader()
    kfbReader.reader.ReadInfo(image, path, Scale, True)

    #获取读取视野倍数
    scale = kfbReader.reader.getReadScale(image)
    # 实例化后，按照说明文档的方法，读取kfb格式文件的Roi区域
    draw = image.ReadRoi(Roi_x, Roi_y, Roi_w, Roi_h, scale=scale)  # 这个sacle将读取的ROI对应到相应倍数上，影响大

    # # 将所有的pos遍历，画在同一张Roi上面
    # for i in range(1, len(corres_json_list)):
    #     Pos_x = corres_json_list[i]['x']
    #     Pos_y = corres_json_list[i]['y']
    #     Pos_w = corres_json_list[i]['w']
    #     Pos_h = corres_json_list[i]['h']
    #     rela_x, rela_y = caculate_relative_position(Roi_x, Roi_y, Pos_x, Pos_y)
    #
    #     draw = cv.rectangle(draw, (rela_x, rela_y), (rela_x + Pos_w, rela_y + Pos_h), (255, 0, 0), 10)#在图像上画出标记框
    cv.imwrite(f"E:/ali_cervical_carcinoma_data/ROI_image/{labels_filename}.jpg", draw)  #保存图像

    end_time = time.time()
    cost_time = end_time - start_time
    total_time = total_time + cost_time
    print(f'The {labels_filename}  done,which cost {cost_time}s')

    return total_time


Scale = 20  # 这个scale未知作用


corres_labels_root = 'E:/ali_cervical_carcinoma_data/corres_labels_0to9'  #由correspongding_ROI_json_maker.py得来

for k in range(0,10):
    #遍历所有阴性病变文件夹
    kfb_image_root = f'E:/ali_cervical_carcinoma_data/pos_{k}'
    #以kfb文件为基准设置循环
    all = os.walk(kfb_image_root)
    for kfb_root,_,filelist in all:
        for filename in filelist:
            basename_num =filename[:-4].split('_')[1]
            #到corres_labels文件夹中找到对应json 并读取其坐标
            #如果包含filename  如T2019_53.kfb
            labels_all = os.walk(corres_labels_root)
            for _, _, labelslist in labels_all:
                for labels_filename in labelslist:
                    labels_filename_num = labels_filename[:-5].split('_')[1]
                    # 判断json的文件名前几位是否严格等于kfb前几位的文件名，以便全部遍历且一一对应
                    if labels_filename_num == basename_num  :#避免53和530一起被读入图片的情况
                        corres_json_path = os.path.join(corres_labels_root, labels_filename)
                        corres_json_file = open(corres_json_path).read()  # 读取json
                        corres_json_list = json.loads(corres_json_file)  # 将字符串转换为List
                        print(f'\n filename is {filename},labels name is {labels_filename} ,NOW we are at pos_{k}'  )
                        total_time = draw_rectangle(labels_filename,corres_json_list,total_time)


                    else:
                        continue

print(' =  = '*10)
print(f'Total time cost {total_time}s')















