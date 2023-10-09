import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.handocc.backbone import FPN, RGB_Depth
from model.handocc.transformer import Transformer
from model.handocc.regressor import Regressor
class Backbone_HandOcc(nn.Module):
    def __init__(self):
        super(Backbone_HandOcc, self).__init__()
        self.backbone = FPN(pretrained=True)
        self.rgb_depth = RGB_Depth()

    def forward(self, img):
        feature = self.backbone(img)

        return feature


class HandOcc(nn.Module):
    def __init__(self, mode):
        super(HandOcc, self).__init__()
        self.mode = mode
        self.FIT = Transformer(injection=True)
        self.SET = Transformer(injection=False)
        self.regressor = Regressor()

    def forward(self, p_feats, s_feats, targets):

        feats = self.FIT(s_feats, p_feats)
        feats = self.SET(feats, feats)

        if self.mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        else:
            gt_mano_params = None
        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(feats, gt_mano_params)

        return pred_mano_results, gt_mano_results, preds_joints_img


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode):
    # backbone = FPN(pretrained=True)
    FIT = Transformer(injection=True)  # feature injecting transformer
    SET = Transformer(injection=False)  # self enhancing transformer
    regressor = Regressor()
    # depth = DepthNet()
    #
    # if mode == 'train':
    #     FIT.apply(init_weights)
    #     SET.apply(init_weights)
    #     regressor.apply(init_weights)

    model = HandOcc(mode)

    return model