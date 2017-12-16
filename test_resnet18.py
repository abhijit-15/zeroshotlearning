import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from hashtag import hashtags

def get_hashtags(img):
    model = torch.load('/Users/shreyashpandey/acads/cs229/project/models/insta_7.pt')

    labels = ['abstract', 'adventure', 'aerobics', 'animals', 'art', 'baseball', 'beach', 'beauty',
            'birds', 'bright', 'burger', 'cake', 'cars', 'cats', 'cheese', 'church', 'clothes', 
            'cloud', 'concert', 'cricket', 'dance', 'desserts', 'dogs', 'dress', 'electronic_gadgets',
            'face', 'fashion', 'fitness', 'food', 'forests', 'fruits', 'gadgets', 'grayscale', 'gym',
            'hill', 'hotdog', 'icecream', 'indoors', 'landscape', 'library', 'mountain', 'museum', 
            'music', 'night', 'noodles', 'office', 'outdoors', 'outfit', 'pet_animal', 'photography', 
            'pizza', 'portrait', 'pose', 'resort', 'restaurant', 'river', 'room', 'scenery', 'science',
                'sea', 'smartphone', 'soccer', 'sports', 'swimming', 'tech', 'tennis', 'theatre', 
                'travelling', 'trek', 'valley', 'vehicles', 'waterfall', 'weightlifting', 'wildlife', 
                'workout', 'yoga']


    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
    ])

    path_to_img = img
    img_pil = Image.open(path_to_img)

    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    out = model(img_variable)

    #predicted = [labels[x] for x in (out.data.numpy()[0].argsort())[::-1][:10]]


    prob, pred = out.topk(2, 1, True, True)    #top 2
    prob = prob.data.numpy()[0]
    pred = pred.data.numpy()[0]     
    predicted = [labels[x] for x in pred]
    print predicted
    htags = [hashtags[x].split() for x in predicted]
    print htags
    

    prob = prob/prob.sum()
    n_tags = 15

    #print prob
    result = []
    for i,t in enumerate(htags):
        result.append(t[:int(prob[i]*n_tags)+1])

    return result
    #for sampling - eg. numpy.random.choice(numpy.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])

"""test_dir = './train_val/val/'
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
    out  = model(input_var)
    predicted = [labels[x] for x in (out.data.numpy()[0].argsort())[::-1][:10]] 
    
    if predicted[0] == labels[target.numpy()[0]]:
        top1 += 1.0
    if labels[target.numpy()[0]] in predicted[:5]:
        top5 += 1.0
    if labels[target.numpy()[0]] in predicted:
        top10 += 1.0
    print "Image number: ", i, "Top-10 until now: ", top10/(i+1)

top1 /= len(data_loader)
top5 /= len(data_loader)
top10 /= len(data_loader)

print "Top-1 accuracy: ", top1
print "Top-5 accuracy: ", top5
print "Top-10 accuracy: ", top10"""