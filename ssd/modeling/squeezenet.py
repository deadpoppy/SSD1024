import torchvision.models as  models
import torch

def addsqueeze():
    prenet=models.squeezenet1_0(pretrained=False)
    layers=[]
    layers.append(prenet.features[0:11])
    layers.append(prenet.features[11:])
    return layers