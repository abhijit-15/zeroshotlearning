import os
from shutil import copyfile

path = 'data/'

classes = [x for x in os.listdir(path) if os.path.isdir(path+x)] 

train_dst = 'train_val/train/'
val_dst = 'train_val/val/'

for c in classes:
    if not os.path.exists(train_dst+c):
        os.makedirs(train_dst+c)
        os.makedirs(val_dst+c)
    class_dir = path+c+'/'
    imgs = os.listdir(class_dir)
    n_imgs = len(imgs)
    n_train = int(.8*n_imgs)
    train = imgs[:n_train]
    val = imgs[n_train:]
    for i in train:
        copyfile(class_dir+i, train_dst+c+'/'+i)
    for i in val:
        copyfile(class_dir+i, val_dst+c+'/'+i)

    