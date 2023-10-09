"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.idisc.models.defattn_decoder import MSDeformAttnPixelDecoder
from model.idisc.models.fpn_decoder import BasePixelDecoder
from model.idisc.models.id_module import AFP, ISD


class IDisc(nn.Module):
    def __init__(
        self,
        afp: nn.Module,
        pixel_decoder: nn.Module,
        isd: nn.Module,
        afp_min_resolution=1,
        eps: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.afp = afp
        self.pixel_decoder = pixel_decoder
        self.isd = isd
        self.afp_min_resolution = afp_min_resolution
        self.point = nn.Sequential(nn.Conv2d(3, 1, kernel_size=1),
                                   nn.BatchNorm2d(num_features=1),
                                   nn.SiLU(),
                                   nn.AdaptiveMaxPool2d((3, 21)))

    def invert_encoder_output_order(
        self, xs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(
        self, decoder_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(decoder_outputs[self.afp_min_resolution :])

    def forward(
        self,
        encoder_outputs
    ):

        encoder_outputs = self.invert_encoder_output_order(encoder_outputs)

        # DefAttn Decoder + filter useful resolutions (usually skip the lowest one)
        fpn_outputs, decoder_outputs = self.pixel_decoder(encoder_outputs)

        decoder_outputs = self.filter_decoder_relevant_resolutions(decoder_outputs)

        idrs = self.afp(decoder_outputs)
        idrs = tuple((i.reshape(i.shape[0], 1, i.shape[1], i.shape[2]) for i in idrs))
        idrs = torch.cat(idrs, dim=1)
        point = self.point(idrs)
        point = torch.squeeze(point, 1)
        point = torch.tanh(point)

        return point

    def normalize_normals(self, norms):
        min_kappa = 0.01
        norm_x, norm_y, norm_z, kappa = torch.split(norms, 1, dim=1)
        norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0 + 1e-6)
        kappa = F.elu(kappa) + 1.0 + min_kappa
        norms = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
        return norms

    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )
        self.load_state_dict(new_state_dict)

    def get_params(self, config):
        backbone_lr = config["model"]["pixel_encoder"].get(
            "lr_dedicated", config["training"]["lr"] / 10
        )
        params = [
            {"params": self.pixel_decoder.parameters()},
            {"params": self.afp.parameters()},
            {"params": self.isd.parameters()},
            {"params": self.pixel_encoder.parameters()},
        ]
        max_lrs = [config["training"]["lr"]] * 3 + [backbone_lr]
        return params, max_lrs

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config: Dict[str, Dict[str, Any]]):
        pixel_encoder_img_size = config["model"]["pixel_encoder"]["img_size"]
        pixel_encoder_pretrained = config["model"]["pixel_encoder"].get(
            "pretrained", None
        )
        config_backone = {"img_size": np.array(pixel_encoder_img_size)}
        if pixel_encoder_pretrained is not None:
            config_backone["pretrained"] = pixel_encoder_pretrained
        import importlib

        pixel_encoder_embed_dims = [64, 128, 256, 512]
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims

        pixel_decoder = (
            MSDeformAttnPixelDecoder.build(config)
            if config["model"]["attn_dec"]
            else BasePixelDecoder.build(config)
        )
        afp = AFP.build(config)
        isd = ISD.build(config)


        return deepcopy(
            cls(
                pixel_decoder=pixel_decoder,
                afp=afp,
                isd=isd,
                afp_min_resolution=len(pixel_encoder_embed_dims)
                - config["model"]["isd"]["num_resolutions"],
            )
        )

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
                "name": "swin_tiny",
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
    model = IDisc.build(dic).to('cuda')
    img = torch.rand((1, 3, 256, 256)).to('cuda')

    b = model(img)