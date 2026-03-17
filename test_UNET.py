import numpy as np
import os
import cv2
from network.UNet import UNetOri
import torch
from timeit import default_timer as timer
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#选择什么样的网络 1代表UNET，2代表DeepLabV3，3代表PIDNET（loss不一样）
NET_FLAG = 1
strPic = '.jpg'#图像后缀名
W =  int(1344)
H =  int(960)
dataName = 'cup'
filepath = r'./dataset/' + dataName + '/image/'  # [CHG] 测试图像路径
modelPath = './run' + '/test/model_best.pth.tar'  # [CHG] 模型路径
savedir = './dataset/' + dataName + '/rst/'  # [CHG] 存储推理结果图像路径
valTxt = r'./dataset/' + dataName + '/val.txt'
def main():
    if NET_FLAG == 1:
        model = UNetOri().cuda()


    dict_torch = torch.load(modelPath)
    model.load_state_dict(dict_torch['state_dict'])
    model.eval()


    int_num = 0
    for line in open(valTxt, 'r'):# [CHG] val.txt路径
        if line[-1] == '\n':
            line = line[:-1]
        img_name = line.split(' ')[0]
        print(filepath + img_name + strPic)
        s_img = cv2.imread(filepath + img_name + strPic,1)

        [s_h, s_w, _] = s_img.shape
        s_img_bak = s_img.copy()
        img = cv2.resize(s_img, (W, H), interpolation=cv2.INTER_CUBIC)
        s_img_bak = cv2.resize(s_img_bak, (W, H), interpolation=cv2.INTER_CUBIC)
        s_img_bak_R = s_img_bak[:,:,1]

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
        if NET_FLAG == 3:
            output = F.interpolate(output, size=[W,H], mode='bilinear', align_corners=True)
        output = torch.sigmoid(output)
        pred = output.data.cpu().numpy()
        # 进行argmax或者加阈值完成最后的分割
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.astype(dtype='uint8')
        pred = pred[0, :, :]

        mask_tmp = -1 * (pred - 1)
        s_img_bak[:, :, 0] = s_img_bak[:, :, 0] * mask_tmp
        s_img_bak[:, :, 1] = s_img_bak[:, :, 1] * mask_tmp
        s_img_bak[:, :, 2] = s_img_bak[:, :, 2] * mask_tmp + pred*255
        end_time = timer()
        print(end_time - start_time)
        pred = output.cpu().numpy()[0]
        s_img_bak = cv2.resize(s_img_bak, (s_w, s_h), interpolation=cv2.INTER_CUBIC)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        savepath = savedir + img_name + '.png'
        cv2.imwrite(savepath, s_img_bak)

if __name__ == "__main__":
    main()
















