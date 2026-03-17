from utils.dataloaders.datasets import pascal
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    train_set = pascal.VOCSegmentation(args, split='train')#遍历训练图片和label名称
    val_set = pascal.VOCSegmentation(args, split='val')
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)#shuffle打乱顺序
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False, **kwargs)
    test_loader = None

    return train_loader, val_loader, test_loader, num_class

