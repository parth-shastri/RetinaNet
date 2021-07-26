import torch
from torch import nn
import numpy
from torch.utils.data import DataLoader
from torchvision import datasets, models

device = "cuda"

retina_net = models.detection.retinanet_resnet50_fpn(pretrained=True)

num_classes = 2


if torch.cuda.is_available():
    retina_net.to(device)

