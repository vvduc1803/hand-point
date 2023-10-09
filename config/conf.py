import os
import os.path as osp
import sys

class Config:
    ## dataset
    # HO3D, DEX_YCB
    trainset = 'DEX_YCB'
    testset = 'DEX_YCB'
    architecture = "hand_point"

    ## input shape
    input_img_shape = (256, 256)

    ## training config
    if trainset == 'HO3D':
        lr_dec_epoch = [10 * i for i in range(1, 7)]
        end_epoch = 70
        lr = 1e-5
        lr_dec_factor = 0.7
    elif trainset == 'DEX_YCB':
        lr_dec_epoch = [i for i in range(1, 999)]
        end_epoch = 999
        lr = 2e-4
        lr_dec_factor = 0.8

    batch_size = 2  # per GPU
    lambda_mano_verts = 1e4
    lambda_mano_joints = 1e4
    lambda_mano_pose = 10
    lambda_mano_shape = 0.1
    lambda_joints_img = 100

    ## others
    num_thread = -1
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join(root_dir, 'utils', 'manopth')
    dic = {
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

    def set_args(self, gpu_ids, batch, num, epoch, continue_train):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        self.batch_size = batch
        self.num_thread = num
        self.end_epoch = epoch
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder

add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

