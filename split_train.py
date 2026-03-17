import os
import random
total_list = list()
for file in os.listdir(r'./dataset/cup/image/'):
    total_list.append(file)
random.shuffle(total_list)
val_percent = 0.1
train_percent = 0.9    # 这两行设置训练集的比例

ftrain_new = open('./dataset/cup/train.txt', 'w')
fval_new = open('./dataset/cup/val.txt', 'w')

train_len = int(total_list.__len__()*train_percent)
for line in total_list[0:train_len]:
    ftrain_new.write(line.split('.')[0] + '\n')
for line in total_list[train_len:total_list.__len__()]:
    fval_new.write(line.split('.')[0] + '\n')
# for line in total_list[0:train_len]:
#     ftrain_new.write(line + '\n')
# for line in total_list[train_len:total_list.__len__()]:
#     fval_new.write(line + '\n')
# for line in total_list[0:train_len]:
#     ftrain_new.write(path + line + '     ' + path + line.split('.')[0]+'_label.png' + '\n')
# for line in total_list[train_len:total_list.__len__()]:
#     fval_new.write(path + line + '     ' + path + line.split('.')[0]+'_label.png' + '\n')