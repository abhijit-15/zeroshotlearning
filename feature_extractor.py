import torch
import torch.nn as nn
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


model = models.alexnet(pretrained=True)

# remove last fully-connected layer
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image_dir = './'  #path to images
data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(image_dir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

for i, (input, target) in enumerate(data_loader):
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        print output.data.numpy().shape
        