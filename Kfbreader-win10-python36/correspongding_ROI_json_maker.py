import time
import json
import os

start_time = time.time()
def paid_time(start_time,end_time):
    paid_time = end_time - start_time
    return print(f'it cost{paid_time}s ')

def judge_inRoi(name):
    json_path = f'E:/ali_cervical_carcinoma_data/labels/{name}.json'
    print('json_path is: ',json_path)
    print(f'reading filename is pos_0/{name}.kfb \n')
    json_file = open(json_path).read()
    json_list = json.loads(json_file) #字符串转为List里面包含一个字典

    # 将pos和roi坐标分类
    pos_list =[]
    roi_list =[]
    for i in range(0,len(json_list)):

        if json_list[i]['class'] == 'roi':

            roi_list.append(json_list[i])
        elif json_list[i]['class'] == 'pos':

            pos_list.append(json_list[i])
        else:
            print('there are something wrong')
            continue

    #计算右下角
    for i in range(0,len(roi_list)):
        corres_list = []
        corres_list.append(roi_list[i])
        Roi_range_x = roi_list[i]['x']
        Roi_range_y = roi_list[i]['y']
        Roi_range_w = roi_list[i]['w']
        Roi_range_h = roi_list[i]['h']
        rightpoint_x = Roi_range_x +Roi_range_w
        rightpoint_y = Roi_range_y +Roi_range_h
        for j in range(0,len(pos_list)):
            pos_range_x = pos_list[j]['x']
            pos_range_y = pos_list[j]['y']
            #判断某个POS是否在当前循环的ROI内部
            if Roi_range_x<pos_range_x<rightpoint_x and Roi_range_y<pos_range_y<rightpoint_y:
                corres_list.append(pos_list[j])
        jsondata = json.dumps(corres_list)
        f = open(os.path.join(r'E:/ali_cervical_carcinoma_data/corres_labels_0to',f'{name}_Roi{i}.json'),'w')
        f.write(jsondata)
        f.close()

scale = 20

for number in range(0,10):
    root = f'E:/ali_cervical_carcinoma_data/pos_{number}'
    all = os.walk(root)
    for _,_,filelist in all:
        for filename in filelist:
            name = filename[:-4]
            path = f'E:/ali_cervical_carcinoma_data/pos_{number}/{name}.kfb'
            judge_inRoi(name)

#不直接输出数值，将pos_0内的ROI和Pos都分开成了若干份json

end_time = time.time()
paid_time(start_time,end_time)
