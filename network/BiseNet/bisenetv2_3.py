
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as Tr

def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
#            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
 #           ConvBNReLU(64, 64, 3, stride=1),
#            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
#            ConvBNReLU(128, 128, 3, stride=1),
 #           ConvBNReLU(128, 128, 3, stride=1),
        )
    def forward(self, x):#Full的
        feat0 = self.S1(x)
        feat1 = self.S2(feat0)
        feat = self.S3(feat1)
        return feat0, feat1, feat#BiSeNetV2_2_Full 网络需要浅层信息


class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)#通道数变为128
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        # feat = torch.mean(x, dim=(2, 3), keepdim=True)
        #feat = nn.AdaptiveAvgPool2d(output_size=[1, 1])(x)#不管输入图像是多大，输出图像都是1*1大小
        feat = x #[CHG] BY TF WIN7版本的onnx不支持AdaptiveAvgPool2d操作
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=1):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=1):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
          #  GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
           # GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
           # GELayerS1(128, 128),
            #GELayerS1(128, 128),
          #  GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):
    def __init__(self,row = 800,col = 1728):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        #----------------轮胎------------------------#
        self.up1 = nn.Upsample(size=[int(row/8), int(col/8)])#输入h/8,w/8
        self.up2 = nn.Upsample(size=[int(row/8), int(col/8)])
       # self.up1 = nn.Upsample(size=[64, 128])#输入h/8,w/8
        #self.up2 = nn.Upsample(size=[64, 128])
        # self.up1 = nn.Upsample(size=[100, 216])
        # self.up2 = nn.Upsample(size=[100, 216])
        # self.up1 = nn.Upsample(size=[72, 28])
        # self.up2 = nn.Upsample(size=[72, 28])
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        # dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        # print(right1.shape,left1.shape)
        # print(right2.shape, left2.shape)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out

class SegmentHead_Full(nn.Module):

    def __init__(self, in_chan, n_classes=1):
        super(SegmentHead_Full, self).__init__()
        self.conv_out = nn.Conv2d(in_chan, n_classes, 1)

    def forward(self, x):
        feat = self.conv_out(x)
        return feat

class upsampleLayer(nn.Module):

    def __init__(self, in_chan, out_chan,H,W):
        super(upsampleLayer, self).__init__()
        self.conv0 = ConvBNReLU(64, out_chan, 3, stride=1)
        self.conv1 = ConvBNReLU(in_chan, out_chan, 1,padding=0, stride=1)
        self.up = nn.Upsample(size=[int(H/2), int(W/2)])
        # self.up = nn.UpsamplingNearest2d(size=[int(H/2), int(W/2)])
        # self.up = nn.ConvTranspose2d(out_chan, out_chan, 4, 4)
        self.conv2 = ConvBNReLU(out_chan, out_chan, 3, stride=1)

        # self.conv3 = ConvBNReLU(out_chan, out_chan, 3, stride=1)

    def forward(self, x,detail):
        detail = self.conv0(detail)
        feat = self.conv1(x)
        # print(feat.shape)
        feat = self.up(feat)
        # print(feat.shape)
        feat = self.conv2(feat + detail)
        # feat = self.conv3(feat)
        return feat

class BiSeNetV2_3(nn.Module):

    def __init__(self, n_classes=1, h=1024,w=1024,output_aux=True):
        super(BiSeNetV2_3, self).__init__()
        layer = 8
        self.output_aux = output_aux
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer(h,w)
        self.upsampleLayer = upsampleLayer(128, layer, h,w)
        self.h = h
        self.w = w
        #self.affinity_attention = AffinityAttention(128)
        # ## TODO: what is the number of mid chan ?
        self.head = SegmentHead_Full(layer, n_classes=n_classes)
        if self.output_aux:
            self.aux2 = SegmentHead_Full(16,n_classes=n_classes)
            self.aux3 = SegmentHead_Full(32,n_classes=n_classes)
            self.aux4 = SegmentHead_Full(64,n_classes=n_classes)
            self.aux5_4 = SegmentHead_Full(128,n_classes=n_classes)

        self.init_weights()

    def forward(self, x):
        '''
        feat0   2*64*96*512
        feat1   2*64*48*256
        feat_d  2*128*24*128
        feat_head 2*128*24*128
        '''
        feat0,feat1,feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
       # feat_s = self.affinity_attention(feat_s)#[ADD] 加入注意力机制
        feat_head = self.bga(feat_d, feat_s)
        feat_head = self.upsampleLayer(feat_head,feat0)

        logits = self.head(feat_head)
        if self.output_aux:
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        return logits

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

if __name__ == "__main__":

     x1 = torch.randn(1, 3, 640, 640)#光学
     bga = BiSeNetV2_3(1,640,640,False)
     out1= bga(x1)
     print(out1.size())