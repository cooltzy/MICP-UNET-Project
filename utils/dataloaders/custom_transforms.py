import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Label_smoothing(object):
    """进行标签平滑
    Args:
        alpha；平滑系数,一般根据类别数设置，这里只有一类，故直接指定alpha
    """
    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        mask = mask * (2 * self.alpha - 1)
        mask = mask + (1 - self.alpha)
        return {'image': img,
                'label': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        #print(img.shape,mask.shape)
        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask,1)
        return {'image': img,
                'label': mask}


class RandomGaussinBlur(object):
    
    def __call__(self,sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            kernel_size = (3,3)
            sigma = random.random()
            img = cv2.GaussianBlur(img, kernel_size, sigma)
        return {'image': img,
                'label': mask}


class RandomRotate(object):
    
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            [h, w, _] = img.shape
            angle = np.random.randint(0,360)
            center = (w//2, h//2)
            scale = 1.0
            M = cv2.getRotationMatrix2D(center, angle, scale)
            img = cv2.warpAffine(img, M, (w,h))
            mask = cv2.warpAffine(mask, M, (w,h))
        return {'image': img,
                'label': mask}


class RandomCrop(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        scale_h = img.shape[0]
        scale_w = img.shape[1]
        if random.random() < 0.3:
            bbox_h = np.random.randint(0, scale_h, 2)
            bbox_w = np.random.randint(0, scale_w, 2)
            bbox = []
            for num in bbox_h:
                bbox.append(num)
            for num in bbox_w:
                bbox.append(num)
            bbox = np.array(bbox)
            [xmin, ymin, xmax, ymax] = np.sort(bbox)
            img_temp = np.zeros([scale_h,scale_w,3], dtype=np.float32)
            mask_temp = np.zeros([scale_h,scale_w], dtype=np.float32)
            img_temp[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]
            mask_temp[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
            img = img_temp
            mask = mask_temp
        return {'image': img,
                'label': mask}

class RandomHSV(object):
    """Generate randomly the image in hsv space."""
    def __init__(self, h_r, s_r, v_r):
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def __call__(self, sample):
        # print(1)
        if np.random.randint(0, 10, 1)[0] >= 5:
            image = sample['image']
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0].astype(np.int32)
            s = hsv[:, :, 1].astype(np.int32)
            v = hsv[:, :, 2].astype(np.int32)
            # delta_h = np.random.randint(-self.h_r, self.h_r)
            # delta_s = np.random.randint(-self.s_r, self.s_r)
            delta_h = 0
            delta_s = 0
            delta_v = np.random.randint(-self.v_r, 10)
            h = (h + delta_h) % 180
            s = s + delta_s
            s[s > 255] = 255
            s[s < 0] = 0
            v = v + delta_v
            v[v > 255] = 255
            v[v < 0] = 0
            hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.uint8)
            sample['image'] = image
        return sample

class RandomCUTOUT(object):
    def __call__(self, sample):
        # print(1)
        if np.random.randint(0, 10, 1)[0] >= 3:
            image = sample['image']
            ret, label = cv2.threshold(image[:, :, 0], 0, 255, cv2.THRESH_OTSU)
            label[label > 0] = 1
            label_ = 1 - label
            label_ = label_ * np.random.randint(0, 255, 1)[0]
            image[:, :, 0] = image[:, :, 0] * label + label_
            image[:, :, 1] = image[:, :, 1] * label + label_
            image[:, :, 2] = image[:, :, 2] * label + label_
            sample['image'] = image
        return sample

class RandomScaling(object):
    '''random crop and reshape for multi size feature'''
    def __init__(self, min_scale_factor, max_scale_factor, scale_factor_step_size):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        scale_h = img.shape[0]
        scale_w = img.shape[1]
        #
        if np.random.randint(0, 10, 1)[0] >= 7:
            if self.min_scale_factor < 0 or self.min_scale_factor > self.max_scale_factor:
                raise ValueError('Unexpected value of min_scale_factor.')
            # step_size != 0
            scale_factors = np.arange(self.min_scale_factor, self.max_scale_factor + 0.1, self.scale_factor_step_size)
            np.random.shuffle(scale_factors)
            shuffled_scale_factors = scale_factors[0]
            img = cv2.resize(img, None, fx=shuffled_scale_factors, fy=shuffled_scale_factors)
            mask = cv2.resize(mask, None, fx=shuffled_scale_factors, fy=shuffled_scale_factors)

            # Pad image and label
            if shuffled_scale_factors < 1.0:
                # print(1)
                pad_h_top = int((scale_h - img.shape[0]) / 2)
                pad_h_bot = (scale_h - img.shape[0]) - pad_h_top
                pad_w_l = int((scale_w - img.shape[1]) / 2)
                pad_w_r = (scale_w - img.shape[1]) - pad_w_l
                img = cv2.copyMakeBorder(img, pad_h_top, pad_h_bot,
                                         pad_w_l, pad_w_r,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
                mask = cv2.copyMakeBorder(mask, pad_h_top, pad_h_bot,
                                          pad_w_l, pad_w_r,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Randomly crop the image and label.
            if shuffled_scale_factors > 1.0:
                # print(2)
                crop_offset_h = np.random.randint(0, img.shape[0] - scale_h + 1)
                crop_offset_w = np.random.randint(0, img.shape[1] - scale_w + 1)
                img = img[crop_offset_h:crop_offset_h + scale_h, crop_offset_w:crop_offset_w + scale_w]
                mask = mask[crop_offset_h:crop_offset_h + scale_h, crop_offset_w:crop_offset_w + scale_w]

        return {'image': img,
                'label': mask}





