# model_utils.py
import torch
import torch.nn as nn
from collections import OrderedDict

def custome_X3D(num_classes):
    model_name = 'x3d_m'
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    input_size = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features=input_size, out_features=num_classes)
    return model

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    return model
