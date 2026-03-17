import torch
import torch.nn.functional as F
import torch.nn as nn
#from .unet_parts import inconv, down, up, outconv
from .unet_parts import inconv, down, up, outconv

#--------------------------------------------#
# 适合32 * 32 大小的图像
class UNet32(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet32, self).__init__()
        self.inc = inconv(n_channels, 128)#两次卷积
        self.down1 = down(128, 256)#下采样+两次卷积
        self.down2 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 128)
        self.outc = outconv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        # x = F.softmax(x)
        # x = torch.sigmoid(x)
        # print(x.shape)
        # x = torch.argmax(x,dim=1)
        return x
#
#
class UNetOri(nn.Module):
    def __init__(self, n_channels=3, n_classes=1,sigmoid = False,softMax = False):#默认值
        super(UNetOri, self).__init__()
        self.sigmoid = sigmoid
        self.softMax = softMax
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x) # 2 64 256 256
        x2 = self.down1(x1)# 2 128 128 128
        x3 = self.down2(x2)# 2 256 64 64
        x4 = self.down3(x3)# 2 512 32 32
        x5 = self.down4(x4)# 2 512 16 16
        x = self.up1(x5, x4)# 输入 2 512 16 16 ， 2 512 32 32，输出 2 256 32 32
        x = self.up2(x, x3)# 输入 2 256 32 32 ， 2 256 64 64，输出 2 128 64 64
        x = self.up3(x, x2)# 输入 2 128 64 64  ， 2 128 128 128，输出 2 64 128 128
        x = self.up4(x, x1)# 输入 2 64 128 128  ， 2 64 256 256，输出 2 64 256 256
        x = self.outc(x)
        # x = F.softmax(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        elif self.softMax:
            x = torch.argmax(torch.softmax(x, dim=1), dim=1)
        else:
            return x
       # x = torch.sigmoid(x)
        # print(x.shape)
        # x = torch.argmax(x,dim=1)
        #return x


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

        self.x_down = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)  ##上采样到与x分辨率相同
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
            ##dim=1:逐通道相加，假设x_k * y_q的shape为[4, 32, 32, 64]，相加后shape变为[4, 32, 64]，再通过unsqueeze(1)升维为[4, 1, 32, 64]

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)  ##上采样到与x分辨率相同
        x = (1 - sim_map) * x + sim_map * y
        #x = self.x_down(x) #通道数降低为原来的一半
        return x

class UNetNEW(nn.Module):
    def __init__(self, n_channels=3, n_classes=1,sigmoid = False,lossFlag = False):#默认值
        super(UNetNEW, self).__init__()
        self.sigmoid = sigmoid
        self.lossFlag = lossFlag
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.PagFM_up1 = PagFM(512, 256)
        self.PagFM_up2 = PagFM(256, 128)
        self.PagFM_up3 = PagFM(128, 64)
        self.PagFM_up4 = PagFM(64, 32)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)

        self.outc_up1 = outconv(256, n_classes)
        self.outc_up2 = outconv(128, n_classes)
        self.outc_up3 = outconv(64, n_classes)
        self.outc = outconv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)# 2 64 256 256
        x2 = self.down1(x1) # 2 128 128 128
        x3 = self.down2(x2) # 2 256 64 64
        x4 = self.down3(x3)# 2 512 32 32
        x5 = self.down4(x4) # 2 512 16 16
        x_up1 = self.PagFM_up1(x4,x5) #输入 2 512 32 32 和 2 512 16 16 输出 2 512 32 32
        x_up1 = self.up1(x5,x_up1) #输入 2 512 16 16 和 2 512 32 32 输出 2 256 32 32
        x_up2 = self.PagFM_up2(x3,x_up1)
        x_up2 = self.up2(x3, x_up2)
        x_up3 = self.PagFM_up3(x2,x_up2)
        x_up3 = self.up3(x2, x_up3)
        x_up4 = self.PagFM_up4(x1,x_up3)
        x_up4 = self.up4(x1, x_up4)
        x = self.outc(x_up4)
        # x = F.softmax(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        if self.lossFlag:
            x_up1 = self.outc_up1(x_up1)
            x_up2 = self.outc_up2(x_up2)
            x_up3 = self.outc_up3(x_up3)
            return x_up1,x_up2,x_up3,x
        else:
            return x


if __name__ == "__main__":
    x1 = torch.randn(2, 3, 700, 700)
    bga = UNetOri(n_channels=3, n_classes=3)
    out1 = bga(x1)
    print(out1.size())
    















