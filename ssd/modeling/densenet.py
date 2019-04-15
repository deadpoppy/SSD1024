import torch
import torchvision.models as models


def adddensenet():
    prenet=models.densenet161(pretrained=True)
    layers=[]
    feayure1=prenet.features[0:6]
    for parames in feayure1[0:4].parameters():
        for ten in parames:
            ten=ten.detach()
    layers.append(feayure1)
    layers.append(prenet.features[6:8])
    layers.append(prenet.features[8:10])
    # for parame in layers:
    #     for param in parame.parameters():
    #         param.requires_grad = False
    return layers
# net=models.densenet161(pretrained=False)
# print(net.features)