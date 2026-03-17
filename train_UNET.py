import argparse
import os
from network.UNet import UNetOri
from network.BiseNet.bisenetv2_3 import BiSeNetV2_3
import torch.nn.functional as F
import torch
from utils.saver import Saver
from utils.dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from utils.summaries import TensorboardSummary
import torch.nn as nn
from torch.nn import init
from utils.lr_scheduler import LR_Scheduler, Cosine_Annealing
from tqdm import tqdm
import numpy as np
from draw import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#选择什么样的网络 1代表UNET，2代表DeepLabV3，3代表PIDNET（loss不一样）
NET_FLAG = 2
dataName = 'cup'#数据文件夹名称
checkname = ''#模型保存名称
trainWidth =  int(4080)
trainHei =  int(2048)
batchSize = 4
numClasses = 1
epochs = 200
lr = 0.1
if NET_FLAG==1:
    checkname = 'UNET'
elif NET_FLAG==2:
    checkname = 'BiSeNetV2_3'


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)#返回训练时模型存储路径
        self.saver.save_experiment_config()#将一些训练参数写入txt文件
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if NET_FLAG == 1:
            model = UNetOri()
        elif NET_FLAG ==2:
            model = BiSeNetV2_3(n_classes=numClasses, h=trainHei,w=trainWidth,output_aux=True)

        for m in model.modules():#model.modules()会遍历所有的子层
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()

        optimizer = torch.optim.SGD(model.parameters(),#SGD计算梯度
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)#计算LOSS
        self.model, self.optimizer = model, optimizer
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader), warmup_epochs=5)#学习策略

        if args.cuda:
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        self.best_pred = 0.0

        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O0', verbosity=0)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        if args.resume is not None:#判断是否存在预训练模型
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)['state_dict']
            model_static = model.state_dict()
            pretrained_static = {k: v for k, v in checkpoint.items() if
                                 (k in model_static and checkpoint[k].size() == model_static[k].size())}
            model_static.update(pretrained_static)
            if args.cuda:
                print("loading model")
                self.model.module.load_state_dict(model_static)
            else:
                self.model.load_state_dict(model_static)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader, ncols=5)#显示进度条
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)#学习策略
            self.optimizer.zero_grad()#梯度初始化
            if NET_FLAG == 1:
                output = self.model(image)
                loss = self.criterion(output, target)#计算loss值
            elif NET_FLAG == 2:
                output,output1,output2,output3,output4 = self.model(image)#返回的是8倍下采样的
                output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=True)
                output1 = F.interpolate(output1, size=target.shape[1:], mode='bilinear', align_corners=True)
                output2 = F.interpolate(output2, size=target.shape[1:], mode='bilinear', align_corners=True)
                output3 = F.interpolate(output3, size=target.shape[1:], mode='bilinear', align_corners=True)
                output4 = F.interpolate(output4, size=target.shape[1:], mode='bilinear', align_corners=True)
                loss = self.criterion(output, target) + self.criterion(output1, target)+self.criterion(output2, target)+self.criterion(output3, target)+self.criterion(output4, target) 
           
            if mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.6f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch+1, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.6f' % train_loss)

        if self.args.no_val:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename='checkpoint.pth.tar')
        return train_loss

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, ncols=5, desc='\r')
        test_loss = 0.0
        total_Dice = list()
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if NET_FLAG == 1:
                    output = self.model(image)
                    loss = self.criterion(output, target)#计算loss值
                elif NET_FLAG == 2:
                    output,output1,output2,output3,output4 = self.model(image)#返回的是8倍下采样的
                    output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=True)
                    output1 = F.interpolate(output1, size=target.shape[1:], mode='bilinear', align_corners=True)
                    output2 = F.interpolate(output2, size=target.shape[1:], mode='bilinear', align_corners=True)
                    output3 = F.interpolate(output3, size=target.shape[1:], mode='bilinear', align_corners=True)
                    output4 = F.interpolate(output4, size=target.shape[1:], mode='bilinear', align_corners=True)
                    loss = self.criterion(output, target) + self.criterion(output1, target)+self.criterion(output2, target)+self.criterion(output3, target)+self.criterion(output4, target) 
            test_loss += loss.item()
            tbar.set_description('Test loss: %.6f' % (test_loss / (i + 1)))

            output = torch.sigmoid(output)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred.astype(dtype='uint8')
            input_flat = pred.flatten()#返回折叠成一个一维的数组
            target_flat = target.flatten()
            smooth = 1e-7
            intersection = input_flat * target_flat
            max_target = max(target_flat)
            if max_target == 0:#考虑的是非0值
                continue
            Dice = 2 * (np.count_nonzero(intersection) + smooth) / (
                    np.count_nonzero(input_flat) + np.count_nonzero(target_flat) + smooth)
            total_Dice.append(Dice)

        avg_Dice = np.mean(total_Dice)
        total_Dice.clear()
        self.writer.add_scalar('val/Dice', avg_Dice, epoch+1)
        print('Validation:')
        print('[Epoch: %d, numImages: %6d]' % (epoch+1, i * self.args.batch_size + image.data.shape[0]))
        print("Dice：{}".format(avg_Dice))
        print('Loss: %.6f' % test_loss)

        new_pred = avg_Dice
        if new_pred > self.best_pred:
            # if 1:
            print("save")
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        return avg_Dice

def main():
    parser = argparse.ArgumentParser(description="UNET")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=128,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='crop image size')
    #----------------------------宽高、均值方差、类别----------------------------------#
    parser.add_argument('--dataset', type=str, default=dataName,  # [CHG]修改模型存储位置
                        choices=['UNET'], help='dataset name (default: pascal)')
    parser.add_argument('--height_size', type=int, default=trainHei,#[CHG] 图像宽高
                        help='image height size')
    parser.add_argument('--width_size', type=int, default=trainWidth,#[CHG]
                        help='image width size')
    parser.add_argument('--mean', type=list, default=(0,0,0),
                        help='img mean')
    parser.add_argument('--std', type=list, default=(1.0, 1.0, 1.0),
                        help='img std')
    parser.add_argument('--numClasses', type=int, default = numClasses,  # [CHG]目前只是一类
                        help='numClasses')
    baseDir = r'./dataset/' + dataName + '/'
    parser.add_argument('--base_dir', type=str, default=baseDir,  # [CHG]  #图片路径
                        help='image path')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR',  # [CHG]学习率
                        help='learning rate (default: auto)')
    parser.add_argument('--loss_type', type=str, default='ohem_dice',# [CHG]loss 方法
                        choices=['focal_dice', 'ohem_dice'],
                        help='loss func type (default: ce)')
    parser.add_argument('--batch_size', type=int, default = batchSize,  # [CHG]修改批量处理数据大小
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--resume', type=str, default=None,   #  [CHG] 预训练模型
                        help='put the path to resuming file if needed')
    # ----------------------------------------------------------------#
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N', # [CHG]epoch 训练次数
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkname', type=str, default=checkname,#[CHG]
                        help='set the checkpoint name')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]#选择GPU
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.epochs is None:
        epoches = {
            'pascal': 600,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None: #初始化batchSize
        args.test_batch_size = 1

    if args.lr is None:#初始化学习率
        lrs = {
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:#模型保存路径
        args.checkname = 'KS-' + str(args.backbone)

    print(args)

    torch.manual_seed(args.seed)#随机种子，参数初始化
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    # trainer.validation_multiclass(0)
    plot_x=[];ListTrainLoss=[];ListValDice = []#获取训练精度和测试
    plotPath = os.path.join('run', args.dataset, args.checkname)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        plot_x.append(epoch+1)
        trainLoss = trainer.training(epoch)
        ListTrainLoss.append(trainLoss)
        if (not trainer.args.no_val) and epoch % args.eval_interval == (args.eval_interval - 1):
                valDice = trainer.validation(epoch)
                ListValDice.append(valDice)
    #plotScore(plotPath,plot_x,ListTrainLoss,ListValDice)
    plotScoreLoss(plotPath, plot_x, ListTrainLoss)
    plotScoreDice(plotPath, plot_x, ListValDice)
    print("train ended!")
if __name__ == "__main__":
    main()