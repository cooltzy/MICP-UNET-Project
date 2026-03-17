import numpy as np
import os
import cv2
from network.UNet import UNetOri
import torch
from timeit import default_timer as timer
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
from json import load, dump

#选择什么样的网络 1代表UNET，2代表DeepLabV3，3代表PIDNET（loss不一样）
NET_FLAG = 1
strPic = '.jpg'#图像后缀名
W =  int(2048)
H =  int(1280)
dataName = 'test'
filepath = r'./dataset/' + dataName + '/image/'  # [CHG] 测试图像路径
modelPath = './run/' + dataName + '/model_best.pth.tar'  # [CHG] 模型路径
def GetJsonRst(binImg):
    objects = []
    category = 'cup'
    group = 1
    layer = 'category'
    bbox = 'category'
    iscrowd = 0 #0代表false
    note = ''
    contours, _ = cv2.findContours(binImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    layerNum = 0
    ListArea = []
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area_c = cv2.contourArea(c)
        if area_c<1500:
            continue
        layerNum = layerNum + 1
        ListArea.append(area_c)
        x, y, w, h = cv2.boundingRect(c)
        binImg[y:y + h, x:x + w] = 0
        c_list = c.tolist()
        segmentation = [b for a in c_list for b in a]
        bbox = [float(x),float(y),float(x+w),float(y+h)]
        my_dict = {}
        my_dict['category'] = category
        my_dict['group'] = group
        my_dict['segmentation'] = segmentation
        my_dict['area'] = area_c
        my_dict['layer'] = layerNum
        my_dict['bbox'] = bbox
        my_dict['iscrowd'] = bool(iscrowd)
        my_dict['note'] = note
        objects.append(my_dict)
    return objects,ListArea

def save_annotation(binImg, img_folder, img_name, label_path,txt_path):
    objects,ListArea = GetJsonRst(binImg)
    #面积写到txt
    fileArea = open(txt_path, 'w')
    for i in range(len(ListArea)):
        lineTxt = ("area : {:2f} , num : {:2d}".format(ListArea[i],i+1))
        fileArea.write(lineTxt)
        fileArea.write('\n')
    sum_area = sum(ListArea)
    lineTxt = ("total area : {:2f} , total num :  {:2d}".format(sum_area,len(ListArea)))
    fileArea.write(lineTxt)
    fileArea.write('\n')
    fileArea.close()


    description = "ISAT"
    width = 5480   #图像宽
    height = 3648  #图像高
    depth = 3
    note = ""
    dataset = {}
    dataset['info'] = {}
    dataset['info']['description'] = description
    dataset['info']['folder'] = img_folder
    dataset['info']['name'] = img_name
    dataset['info']['width'] = width
    dataset['info']['height'] = height
    dataset['info']['depth'] = depth
    dataset['info']['note'] = note
    dataset['objects'] = []
    for obj in objects:
        object = {}
        object['category'] = obj['category']
        object['group'] = obj['group']
        object['segmentation'] = obj['segmentation']
        object['area'] = obj['area']
        object['layer'] = obj['layer']
        object['bbox'] = obj['bbox']
        object['iscrowd'] = obj['iscrowd']
        object['note'] = obj['note']
        dataset['objects'].append(object)
    with open(label_path, 'w') as f:
        dump(dataset, f, indent=4)
    return True

def BatchTest(FilePath,MaskBinFile,MaskOriFile,MaskJsonFile,MaskTxtFile):
    if not os.path.exists(MaskBinFile):
        os.makedirs(MaskBinFile)
    if not os.path.exists(MaskOriFile):
        os.makedirs(MaskOriFile)
    if not os.path.exists(MaskJsonFile):
        os.makedirs(MaskJsonFile)
    if not os.path.exists(MaskTxtFile):
        os.makedirs(MaskTxtFile)
    # 加载模型
    if NET_FLAG == 1:
        model = UNetOri().cuda()
    dict_torch = torch.load(modelPath)
    model.load_state_dict(dict_torch['state_dict'])
    model.eval()

    img_test = (glob.glob(FilePath))
    for index, img_file in enumerate(img_test):
        s_img = cv2.imread(img_file)
        [s_h, s_w, _] = s_img.shape
        s_img_ori = s_img.copy()
        s_img_bin = np.zeros([W, H,1], dtype=np.uint8)#二值图
        s_img_bin = cv2.resize(s_img_bin, (W, H), interpolation=cv2.INTER_CUBIC)
        s_img_ori = cv2.resize(s_img_ori, (W, H), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(s_img, (W, H), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img /= 255.0
        mean = (0, 0, 0)
        std = (1.0, 1.0, 1.0)
        img -= mean
        img /= std
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        start_time = timer()
        with torch.no_grad():
            output = model(img.cuda())
        output = torch.sigmoid(output).squeeze()
        pred = output.data.cpu().numpy()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.astype(dtype='uint8')
        s_img_bin = s_img_bin + pred*255
        s_img_ori[:, :, 0] = s_img_ori[:, :, 0] * pred
        s_img_ori[:, :, 1] = s_img_ori[:, :, 1] * pred
        s_img_ori[:, :, 2] = s_img_ori[:, :, 2] * pred
        end_time = timer()
        print(end_time - start_time)
        s_img_bin = cv2.resize(s_img_bin, (s_w, s_h), interpolation=cv2.INTER_CUBIC)
        s_img_ori = cv2.resize(s_img_ori, (s_w, s_h), interpolation=cv2.INTER_CUBIC)
        img_name = img_file.split('/')[-1]
        print(img_name)
        savepath = MaskBinFile + img_name
        savepathOri = MaskOriFile + img_name
        label_path = MaskJsonFile+img_name.split('.')[0]+'.json'
        txt_path = MaskTxtFile + img_name.split('.')[0] + '.txt'
        #保存标注结果
        save_annotation(s_img_bin, MaskBinFile, img_name, label_path,txt_path)
        cv2.imwrite(savepath, s_img_bin)
        cv2.imwrite(savepathOri, s_img_ori)

if __name__ == "__main__":
    FilePath = './dataset/' + dataName + '/image/*'
    MaskBinFile = './dataset/' + dataName + '/rst_bin/'
    MaskOriFile = './dataset/' + dataName + '/rst_binori/'
    MaskJsonFile = './dataset/' + dataName + '/rst_json/'
    MaskTxtFile = './dataset/' + dataName + '/rst_txt/'
    BatchTest(FilePath,MaskBinFile,MaskOriFile,MaskJsonFile,MaskTxtFile)















