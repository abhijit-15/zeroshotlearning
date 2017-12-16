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



#sentences = word2vec.Text8Corpus('text8')
#w2v = word2vec.Word2Vec(sentences, size=500)
#w2v.save_word2vec_format('text.model.500.bin', binary=True)
#usage - w2v['word'] for 300d word vector


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

#unseen_classes = ['swimming', 'clothes', 'dogs']
vgg16 = models.vgg16(pretrained=True)

w2v = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

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
"""test_dir = './'
data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

print len(data_loader)
for i, (input, target) in enumerate(data_loader):
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)"""
path_to_img = 'images/fashion.jpg'
img_pil = Image.open(path_to_img)

img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)
input_var = Variable(img_tensor)
# compute output
output = vgg16(input_var)

prob, pred = output.topk(10, 1, True, True)   
prob = prob.data.numpy()[0]
pred = pred.data.numpy()[0]     
pred_classes = [classes[x] for x in pred]
"""for i,c in enumerate(pred_classes):
    print prob[i], c"""
print pred_classes
pred_embedding = sum([prob[j]*get_wv(c) for j,c in enumerate(pred_classes)])/ sum([prob[j] for j,c in enumerate(pred_classes) if sum(get_wv(c))!=0])
dist = np.array([np.dot(get_wv(c),pred_embedding)/norm(get_wv(c))/norm(pred_embedding) for c in unseen_classes])
pred_unseen = [unseen_classes[x] for x in dist.argsort()[::-1][:10]]
print "Unseen class is: ", pred_unseen


                
        
        

