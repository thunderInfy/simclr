import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet18

def get_model(args):
    # defining our deep learning architecture
    resnet = resnet18(pretrained=False)

    head = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(resnet.fc.in_features, 100)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100, 100)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(100, 100))
    ]))

    resnet.fc = head

    if args.multiple_gpus:
        resnet = nn.DataParallel(resnet)

    resnet.to(args.device)

    return resnet
