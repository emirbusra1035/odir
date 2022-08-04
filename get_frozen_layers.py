import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
def get_inception():
    net = models.inception_v3(pretrained=True)
    net.AuxLogits.fc = nn.Linear(768, 2)
    net.fc = nn.Linear(2048, 2)
    counter = 0
    for child in net.children():
        if counter < 13:
            print('Frozen Layers')
            print(child)
            for param in child.parameters():
                param.requires_grad = False

        else:
            print('Fine-Tuned Layers')
            print(child)
        counter += 1
    return net


def get_resnet():
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, 2)
    counter = 0
    for child in net.children():
        if counter < 7:
            print('Frozen Layers')
            print(child)
            for param in child.parameters():
                param.requires_grad = False
        else:
            print('Fine-Tuned Layers')
            print(child)
        counter += 1
    return net


def get_vgg():
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(4096, 2)
    for i, child in enumerate(net.children()):
        if i == 0:
            print('Frozen Layers')
            print(child)
            for c in child.parameters():
                c.requires_grad = False

        print('Fine-Tuned Layers')
        print(child)
    return net

print('inception')
get_inception()
print()
print('resnet')
get_resnet()
print()
print('vgg')
get_vgg()
