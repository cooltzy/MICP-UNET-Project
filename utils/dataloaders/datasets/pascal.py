from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils.dataloaders import custom_transforms as tr
import cv2
import json
import random
meanValue = 0.0#从参数中读取
stdValue = 0.0
def min_max_normlization(data,ratio = 255.0):
    min_val = 0
    max_val = np.max(data)
    nor_data = ratio * ((data - min_val)/(max_val - min_val))
    nor_data[nor_data<0] = 0
    return nor_data


class VOCSegmentation(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self._base_dir = args.base_dir#所有数据路径
        self.NUM_CLASSES = args.numClasses  # [ADD]返回值需要
        self._image_dir = os.path.join(self._base_dir, 'image')
        self._cat_dir = os.path.join(self._base_dir, 'label')
        self._img_str = ".jpg"#图像数据格式
        self._label_str = ".png"#标签图数据格式

        if isinstance(split, str):#判断split是否为str类型
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.args = args
        self.H = args.height_size #【CHG】修改宽高
        self.W = args.width_size
        _splits_dir = self._base_dir
        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line +self._img_str)#默认图片都是png格式
                # print(_image)
                _cat = os.path.join(self._cat_dir, line + self._label_str)#label
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
        assert (len(self.images) == len(self.categories))
        # display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        # print(_img.shape, _target.shape)
        sample = {'image': _img, 'label': _target}
        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = cv2.imread(self.images[index],1)
        #_img = min_max_normlization(_img)
        #cv2.imwrite("1.png",_img)
        #img = np.zeros([self.W, self.H,3],dtype=float)
        #img[:,:,0] = _img;img[:,:,1] = _img;img[:,:,2] = _img
        #_img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        _img = cv2.resize(_img, (self.W, self.H))
        #_mask = cv2.imread(self.categories[index],0)#[chg]灰度图读取
        #_mask = cv2.resize(_mask, (self.W, self.H),interpolation=cv2.INTER_NEAREST)#[chg]
        #_mask = _mask[:, :]#[chg]
        _mask = cv2.imread(self.categories[index])  # [chg]灰度图读取
        _mask = cv2.resize(_mask, (self.W, self.H))#[chg]
        _mask = _mask[:, :, 0]#
        _target = _mask

        return _img, _target


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomScaling(min_scale_factor=0.5, max_scale_factor=2, scale_factor_step_size=0.25),  # 多尺度训练
            tr.RandomHorizontalFlip(),
            tr.RandomHSV(h_r=0, s_r=0, v_r=60),  # 随机亮度变换
            tr.RandomHorizontalFlip(),
            tr.RandomGaussinBlur(),
            tr.RandomRotate(),
            tr.Normalize(mean=self.args.mean, std=self.args.std),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=self.args.mean, std=self.args.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

if __name__ == '__main__':
    print(1)