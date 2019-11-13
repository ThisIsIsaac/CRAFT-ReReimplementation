# coding=utf-8
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
import copy


def copyStateDict2(state_dict):
    new_state_dict = copy.deepcopy(state_dict)
    key_to_delete = "module"
    if key_to_delete in new_state_dict.keys():
        new_state_dict.pop(key_to_delete)
    return new_state_dict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

if __name__ == "__main__":
    synth_dict = torch.load('/DATA1/isaac/CRAFT-Reimplemetation/pretrain/SynthText.pth')
    ic15_dict = torch.load('/DATA1/isaac/CRAFT-Reimplemetation/pretrain/SynthText_IC15.pth')
    ic17_dict = torch.load('/DATA1/isaac/CRAFT-Reimplemetation/pretrain/SynthText_IC13_IC17.pth')
    vgg_state_dict = torch.load('/DATA1/isaac/CRAFT-Reimplemetation/pretrain/vgg16_bn-6c64b313.pth')

    new_state_dict = copyStateDict2(vgg_state_dict)

