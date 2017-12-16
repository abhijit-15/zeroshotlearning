import torchvision.models as models
from gensim.models import word2vec
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
from imagenet_classes import classes
import numpy as np
import nltk
from numpy.linalg import norm
from PIL import Image
from torch.autograd import Variable

unseen_classes = ['abstract', 'adventure', 'aerobics', 'animals', 'art', 'baseball', 'beach', 'beauty',
         'birds', 'bright', 'burger', 'cake', 'cars', 'cats', 'cheese', 'church', 'clothes', 
         'cloud', 'concert', 'cricket', 'dance', 'desserts', 'dogs', 'dress', 'electronics',
          'face', 'fashion', 'fitness', 'food', 'forests', 'fruits', 'gadgets', 'grayscale', 'gym',
           'hill', 'hotdog', 'icecream', 'indoors', 'landscape', 'library', 'mountain', 'museum', 
           'music', 'night', 'noodles', 'office', 'outdoors', 'outfit', 'pets', 'photography', 
           'pizza', 'portrait', 'pose', 'resort', 'restaurant', 'river', 'room', 'scenery', 'science',
            'sea', 'smartphone', 'soccer', 'sports', 'swimming', 'tech', 'tennis', 'auditorium', 
            'tourism', 'trek', 'valley', 'vehicles', 'waterfall', 'weightlifting', 'wildlife', 
            'workout', 'yoga']

def get_wv(s):
    s = s.replace(",","")
    words = nltk.word_tokenize(s)
    count = len([w for w in words if w.lower() in w2v])
    if count == 0:
        return np.zeros(300)
    return sum([w2v[w.lower()] for w in words if w.lower() in w2v])/ float(count)

vgg16 = models.vgg16(pretrained=True)

w2v = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

test_dir = './train_val/val/'
data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

print "Total number of images:", len(data_loader)
top1 = 0
top5 = 0
top10= 0
for i, (input, target) in enumerate(data_loader):        
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    output = vgg16(input_var)
    prob, pred = output.topk(10, 1, True, True)   
    prob = prob.data.numpy()[0]
    pred = pred.data.numpy()[0]     
    pred_classes = [classes[x] for x in pred]    
    #print pred_classes
    pred_embedding = sum([prob[j]*get_wv(c) for j,c in enumerate(pred_classes)])/ sum([prob[j] for j,c in enumerate(pred_classes) if sum(get_wv(c))!=0])
    dist = np.array([np.dot(get_wv(c),pred_embedding)/norm(get_wv(c))/norm(pred_embedding) for c in unseen_classes])
    pred_unseen = [unseen_classes[x] for x in dist.argsort()[::-1][:10]]

    #print "Unseen class is: ", pred_unseen
    if pred_unseen[0] == unseen_classes[target.numpy()[0]]:
        top1 += 1.0
    if unseen_classes[target.numpy()[0]] in pred_unseen[:5]:
        top5 += 1.0
    if unseen_classes[target.numpy()[0]] in pred_unseen:
        top10 += 1.0
    print "Image number: ", i, "Top-10 until now: ", top10/(i+1)

top1 /= len(data_loader)
top5 /= len(data_loader)
top10 /= len(data_loader)

print "Top-1 accuracy: ", top1
print "Top-5 accuracy: ", top5
print "Top-10 accuracy: ", top10


    
    
    
