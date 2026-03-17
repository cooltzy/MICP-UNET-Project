import cv2
import os
import json
import numpy as np
from builtins import str#不然字符串不能使用

#标准的标注格式
# regionsName = 'shapes'
# class_name = 'label'
# polygon = 'points'

regionsName = 'objects'
className = 'category'
polygonName = 'segmentation'

def jsonToLabel(filepath,filepath_img,filepath_new):
    for file in os.listdir(filepath):
        # if file.split('_')[0] != 'chongdong':
        #     continue
        img_file = filepath_img+file[0:-4]+'jpg'
        #print(img_file)
        img = cv2.imread(img_file,1)
        # print(img.shape)
        img_w = img.shape[1]
        img_h = img.shape[0]

        fo = open(filepath+file)
        text = fo.read()
        fo.close()
        annotations = json.loads(text)
        regions = annotations.get(regionsName, None)#labels
        if regions is None:
            print('NULL')
        mask = np.zeros([img_h, img_w], dtype='uint8')
        for mark in regions:
            class_name = mark[className] #label
            if class_name == 'cup':
                polygon = mark.get(polygonName)#points
                polygon = np.array(polygon, dtype=np.int32) # (x,y)
                # print(polygon.shape)
                polygon = polygon*np.array([1, 1])
                polygon = polygon.astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1)
            # if class_name == 'cd':
            #     polygon = mark.get('points')
            #     polygon = np.array(polygon, dtype=np.int32) # (x,y)
            #     # print(polygon.shape)
            #     polygon = polygon*np.array([1, 1])
            #     polygon = polygon.astype(np.int32)
            #     cv2.fillPoly(mask, [polygon], 2)
        #print(mask.shape)
        cv2.imwrite(filepath_new + file.split('.')[0] + '.png', mask)

#获取json里面的标签名称
def GetjsonLabel(filepath,filepath_img,filepath_new):
    for file in os.listdir(filepath):
        # if file.split('_')[0] != 'chongdong':
        #     continue
        img_file = filepath_img+file[0:-4]+'jpg'
        print(img_file)
        img = cv2.imread(img_file,1)
        # print(img.shape)
        img_w = img.shape[1]
        img_h = img.shape[0]

        fo = open(filepath+file)
        text = fo.read()
        fo.close()
        annotations = json.loads(text)
        regions = annotations.get('shapes', None)
        if regions is None:
            print('NULL')
        mask = np.zeros([img_h, img_w], dtype='uint8')
        for mark in regions:
            class_name = mark['label']
            if class_name == 'aggregate':
                polygon = mark.get('points')
                polygon = np.array(polygon, dtype=np.int32) # (x,y)
                # print(polygon.shape)
                polygon = polygon*np.array([1, 1])
                polygon = polygon.astype(np.int32)
                cv2.fillPoly(mask, [polygon], 200)
            if class_name == 'mineral powder':
                polygon = mark.get('points')
                polygon = np.array(polygon, dtype=np.int32) # (x,y)
                # print(polygon.shape)
                polygon = polygon*np.array([1, 1])
                polygon = polygon.astype(np.int32)
                cv2.fillPoly(mask, [polygon], 255)
            # if class_name == 'cd':
            #     polygon = mark.get('points')
            #     polygon = np.array(polygon, dtype=np.int32) # (x,y)
            #     # print(polygon.shape)
            #     polygon = polygon*np.array([1, 1])
            #     polygon = polygon.astype(np.int32)
            #     cv2.fillPoly(mask, [polygon], 2)
        print(mask.shape)
        cv2.imwrite(filepath_new + file.split('.')[0] + '.png', mask)

def GetLabelName(json_folder):
    label_counts = {}  # 用于存储不同label的
    # 获取所有JSON文件的文件名
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            if regionsName in data:
                shapes = data[regionsName]
                for shape in shapes:
                    label = shape[className]
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1
    print("Label counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("Label counting completed.")

# 定义一个函数来修改json内容
def modify_json_content(data):
    # 如果data是一个列表
    if isinstance(data, list):
        i = 0
        while i < len(data):
            # 如果条目是一个字典并且其标签为 "air void"，则删除该条目
            if isinstance(data[i], dict) and data[i].get("label") == "air void":
                data.pop(i)
            else:
                modify_json_content(data[i])
                i += 1
    # 如果data是一个字典
    elif isinstance(data, dict):
        # 检查标签名并进行修改
        if "label" in data:
            if data["label"] in ["aggreagte", "aggreagte","aaggregate","air void"]:
                data["label"] = "aggregate"
            elif data["label"] == ["amineral powder","mmineral filler"]:
                data["label"] = "mineral filler"
        # 递归处理字典中的每一个值
        for key in data:
            modify_json_content(data[key])
    return data

def changeJsonName(jsonfile):
    # 获取当前目录下的所有json文件
    json_files = [f for f in os.listdir(jsonfile) if f.endswith('.json')]
    # 遍历每一个json文件
    for json_file in json_files:
        json_path = os.path.join(jsonfile, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        # 修改json内容
        modified_data = modify_json_content(data)
        # 保存修改后的内容到原文件
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(modified_data, file, ensure_ascii=False, indent=4)

    print("所有JSON文件已成功修改!")

if __name__ == "__main__":
    if False:
        json_folder = r"./dataset/cup/json"
        #changeJsonName(json_folder)
        GetLabelName(json_folder)

    if True:
        filepath = r"./dataset/cup/json/"
        filepath_img = r"./dataset/cup/image/"
        filepath_new = r"./dataset/cup/label/"
        jsonToLabel(filepath, filepath_img, filepath_new)

    




