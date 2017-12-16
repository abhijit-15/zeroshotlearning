import os
from shutil import copyfile
from PIL import Image

path = 'train_val/train/'

dest_tr = 'small/train/'
dest_val= 'small/val/'
classes = [x for x in os.listdir(path) if os.path.isdir(path+x)] 

size = 128, 128
for c in classes:
    class_dir = path+c+'/'
    if not os.path.exists(dest_tr+c):
        os.makedirs(dest_tr+c)
        os.makedirs(dest_val+c)
    imgs = os.listdir(class_dir)
    for img in imgs[:70]:
        try:
            im = Image.open(class_dir+img)
            print im.size
            imr = im.resize(size, Image.ANTIALIAS)
            print imr.size
            imr.save(dest_tr+c+'/'+img, "JPEG")
        except IOError:
            print "cannot create thumbnail for", img
    for img in imgs[50:60]:
        try:
            im = Image.open(class_dir+img)
            im.resize(size, Image.ANTIALIAS)
            im.save(dest_val+c+'/'+img, "JPEG")
        except IOError:
            print "cannot create thumbnail for", img

