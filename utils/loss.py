import torch
import torch.nn as nn
from .lovasz_losses import LovaszLoss
import torch.nn.functional as F
import numpy as np


class Dice(nn.Module):
    def __init__(self, logits=False, ohem=False):
        super(Dice, self).__init__()
        self.logits = logits
        self.ohem = ohem

    def forward(self, logit, target):
        if self.logits:
            logit = torch.sigmoid(logit)
        N = target.size(0)
        smooth = 1e-7
        input_flat = logit.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        if self.ohem:
            target_sum = target_flat.sum(1)
            target_sum[target_sum > 0] = 1
            loss = 1 - loss
            loss = loss * target_sum
            loss = loss.sum() / N
        else:
            loss = 1-loss.sum() / N
        return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

class FocalLoss_(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, logits=False, reduce=True):
        super(FocalLoss_, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal' or 'dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'tversky':
            return self.TverskyLoss
        elif mode == 'lovasz_hinge':
            return self.lovasz_hinge
        elif mode == 'bce_dice':
            return self.bce_dice
        elif mode == 'focal_dice':
            return self.focal_dice
        elif mode == 'focal_Tversky':
            return self.focal_Tversky
        elif mode == 'ohem':
            return self.ohem
        elif mode == 'ohem_dice':
            return self.ohem_dice
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        # new
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)  # N,C,H,W => N,C,H*W
            logit = logit.transpose(1, 2)    # N,C,H*W => N,H*W,C
            logit = logit.contiguous().view(-1, logit.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1)
        loss = criterion(logit, target.long())
        if self.batch_average:
            loss /= n
        return loss

    def FocalLoss(self, logit, target):
        loss = FocalLoss_(logits=True)(logit[:, 0, :, :], target)
        return loss
    
    def DiceLoss(self, logit, target, ohem=False):
        loss = Dice(logits=True, ohem=ohem)(logit[:, 0, :, :], target)
        return loss

    def lovasz_hinge(self, logit, targets, weights=None):
        loss = LovaszLoss()(logit[:, 0, :, :], targets) + self.FocalLoss(logit, targets)
        return loss

    def TverskyLoss(self, logit, targets, beta=0.1, logits=False):
        batch_size = targets.size(0)
        loss = 0.0
        if logits:
            logit = torch.sigmoid(logit)
        for i in range(batch_size):
            prob = logit[i]
            ref = targets[i]
            if ref.sum() != 0:
                alpha = 1.0 - beta
                tp = (ref * prob).sum()
                fp = ((1 - ref) * prob).sum()
                fn = (ref * (1 - prob)).sum()
                tversky = tp / (tp + alpha * fp + beta * fn)

                loss = loss + (1 - tversky)
        # if self.batch_average:
        loss /= batch_size
        return loss

    def bce_dice(self, logit, targets):
        bce_loss = nn.BCELoss(reduction='mean')
        bce_out = bce_loss(logit[:, 0, :, :], targets)
        loss = bce_out + self.DiceLoss(logit, targets)
        return loss

    def focal_dice(self, logit, targets):
        bce_out = self.FocalLoss(logit, targets)
        loss = bce_out + self.DiceLoss(logit, targets) #+ self.TverskyLoss(logit[:, 0, :, :], targets, logits=True)
        return loss

    def focal_Tversky(self, logit, targets):
        bce_out = self.FocalLoss(logit, targets)
        loss = bce_out + self.TverskyLoss(logit[:, 0, :, :], targets, logits=True)
        return loss

    def ohem(self, outputs, targets, alpha=1, gamma=2, OHEM_percent=0.005, logits=True):

        if logits:
            outputs = torch.sigmoid(outputs)
        batch_size = targets.size(0)
        ohem_loss = torch.zeros([batch_size])
        ohem_loss = ohem_loss.cuda()
        target_flat = targets.view(batch_size, -1)
        target_sum = target_flat.sum(1)
        target_sum[target_sum > 0] = 1
        target_sum = -1 * (target_sum - 1) + 0
        for i in range(batch_size):
            output = outputs[i].contiguous().view(-1)
            target = targets[i].contiguous().view(-1)

            max_val = (-output).clamp(min=0)
            loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

            # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
            invprobs = F.logsigmoid(-output * (target * 2 - 1))
            focal_loss = alpha * (invprobs * gamma).exp() * loss

            # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
            OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
            ohem_loss[i] = OHEM.mean()
        ohem_loss = ohem_loss * target_sum
        return ohem_loss.mean()

    def ohem_dice(self, logit, targets):
        loss = self.ohem(logit, targets) + self.DiceLoss(logit, targets, ohem=True)
        return loss

if __name__ == "__main__":
    print('ok')
