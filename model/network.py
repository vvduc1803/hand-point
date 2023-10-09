import torch
from torch import nn
from model.handocc_model import Backbone_HandOcc, HandOcc, get_model
from model.idisc_model import IDisc
from model.pointnet2_model import PointNet2
from model.handocc.backbone import RGB_Depth

class Model(nn.Module):
    def __init__(self, dic, mode='train'):
        super(Model, self).__init__()
        self.backbone_handocc = Backbone_HandOcc()
        self.idisc = IDisc.build(dic)

        self.pointnet2 = PointNet2()

        self.rgb_depth = RGB_Depth()

        self.handocc = get_model(mode)

    def forward(self, img, pr):
        feature_point, feature_img = self.backbone_handocc(img)
        point = self.idisc(feature_point)
        xyz, feature_depth = self.pointnet2(point)
        feature_depth = feature_depth.reshape(-1, 32, 32)
        x1, x2 = self.rgb_depth(feature_img, feature_depth)
        pred_mano_results, gt_mano_results, preds_joints_img = self.handocc(x1, x2, pr)
        return point, (pred_mano_results, gt_mano_results, preds_joints_img)

if __name__ == '__main__':

    dic = {
        "generic": {
            "seed": 13
        },
        "training": {
            "n_iters": 50000,
            "batch_size": 8,
            "validation_interval": 200,
            "nsteps_accumulation_gradient": 2,
            "lr": 0.0002,
            "wd": 0.02,
            "div_factor": 1,
            "final_div_factor": 10,
            "loss": {
                "name": "SILog",
                "weight": 10.0
            }
        },
        "data": {
            "crop": "eigen",
            "train_dataset": "YCBVDataset",
            "val_dataset": "YCBVDataset",
            "train_data_root": "train_real",
            "test_data_root": "test",
            "augmentations": {
                "horizontal_flip": 0.5,
                "random_rotation": 5,
                "random_scale": 0.1,
                "random_translation": 0.1,
                "random_brightness": 0.1,
                "random_contrast": 0.1,
                "random_saturation": 0.1,
                "random_gamma": 0.1,
                "random_hue": 0.1,
                "random_sharpness": 0.1,
                "random_posterize": 4,
                "random_solarize": 0.2,
                "rotation_p": 1,
                "scale_p": 1,
                "translation_p": 0,
                "brightness_p": 1,
                "contrast_p": 1,
                "saturation_p": 1,
                "gamma_p": 1,
                "hue_p": 1,
                "sharpness_p": 0,
                "posterize_p": 0,
                "solarize_p": 0,
                "equalize_p": 0,
                "autocontrast_p": 0
            }
        },
        "model": {
            "num_heads": 1,
            "attn_dec": True,
            "activation": "gelu",
            "expansion": 4,
            "output_dim": 1,
            "isd": {
                "latent_dim": 128,
                "num_resolutions": 3,
                "baseline": False,
                "depths": 2,
                "competition": False
            },
            "pixel_decoder": {
                "heads": 4,
                "depths": 4,
                "hidden_dim": 256,
                "anchor_points": 4
            },
            "pixel_encoder": {
                "img_size": [
                    256, 256
                ],
                "name": "resnet18",
                "lr_dedicated": 2e-05
            },
            "afp": {
                "context_low_resolutions_skip": 1,
                "depths": 2,
                "latent_dim": 128,
                "num_latents": 32
            }
        }
    }

    model = Model(dic).to('cuda')
    # a = torch.rand((2, 3, 256, 256)).to('cuda')
    # model(a)
    # from torchsummary import summary
    # print(summary(model, (3, 256, 256)))

    print(torch.mean(model.rgb_depth.attention_module.weight.weight).item())