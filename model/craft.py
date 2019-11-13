"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.torchutil import *
from model.vgg16_bn import vgg16_bn

# imports for post processing
from util import craft_utils, imgproc
import numpy as np


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, use_vgg16_pretrained=True, freeze=False):
        """

        :param vgg_pretrained_path:
        :param freeze: freeze vgg16_bn weights
        """
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained=use_vgg16_pretrained,  freeze=freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature

    def get_vgg16_state_dict(self):
        """

        :return:
        OrderDict of state_dicts, where key is name of layer (ex. "slice1") and value is state_dict
        of the layer.
        """
        state_dicts = OrderedDict()
        state_dicts["slice1"] = self.basenet.slice1.state_dict()
        state_dicts["slice2"] = self.basenet.slice2.state_dict()
        state_dicts["slice3"] = self.basenet.slice3.state_dict()
        state_dicts["slice4"] = self.basenet.slice4.state_dict()
        state_dicts["slice5"] = self.basenet.slice5.state_dict()
        return state_dicts

    def load_vgg16_state_dict(self, state_dicts):
        """
        :param state_dicts: OrderDict of state_dicts, where key is name of layer (ex. "slice1") and value is state_dict
        of the layer.
        """
        self.basenet.slice1.load_state_dict(state_dicts["slice1"])
        self.basenet.slice2.load_state_dict(state_dicts["slice2"])
        self.basenet.slice3.load_state_dict(state_dicts["slice3"])
        self.basenet.slice4.load_state_dict(state_dicts["slice4"])
        self.basenet.slice5.load_state_dict(state_dicts["slice5"])

    #Todo: complete
    def tensor_to_image(self, net_output, text_threshold, link_threshold, low_text, poly):
        score_text = net_output[0, :, :, 0].cpu().data.numpy()
        score_link = net_output[0, :, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        return boxes, polys, ret_score_text

if __name__ == '__main__':
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)
