"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os.path as osp
import cv2

import numpy as np
import torch
import copy

from pycocotools.coco import COCO
from utils.preprocessing import get_bbox, process_bbox

from utils.mano import MANO

mano = MANO()

from torch.utils.data import Dataset
from utils.preprocessing import load_img, augmentation
from utils.transforms import rigid_align

class DEX_YCBVDataset(Dataset):

    def __init__(self, base_path, mode):
        super().__init__()
        self.min_depth = 0.01
        self.max_depth = 256
        self.mode = mode
        self.base_path = base_path
        self.height = 480
        self.width = 640
        self.root_joint_idx = 0

        self.dataset = []
        self.data_split = 'train' if mode == 'train' else 'test'
        self.annot_path = osp.join(self.base_path, 'annotations')

        # load annotations
        self.load_dataset()

        # if self.data_split != 'train':
        self.eval_results = [[],[]] #[mpjpe_list, pa-mpjpe_list]

    def __len__(self):
        """Total number of samples of data."""
        return len(self.dataset)

    def load_dataset(self):
        db = COCO(osp.join(self.annot_path, "DEX_YCB_s0_{}_subset_data.json".format(self.data_split)))

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.base_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            if self.data_split == 'train':
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                joints_coord_img = np.array(ann['joints_img'], dtype=np.float32)
                hand_type = ann['hand_type']

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)

                if bbox is None:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {"img_path": img_path,
                        "img_shape": img_shape,
                        "joints_coord_cam": joints_coord_cam,
                        "joints_coord_img": joints_coord_img,
                        "bbox": bbox,
                        "cam_param": cam_param,
                        "mano_pose": mano_pose,
                        "mano_shape": mano_shape,
                        "hand_type": hand_type}
            else:
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)
                root_joint_cam = copy.deepcopy(joints_coord_cam[0])
                joints_coord_img = np.array(ann['joints_img'], dtype=np.float32)
                hand_type = ann['hand_type']

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
                if bbox is None:
                    bbox = np.array([0, 0, img['width'] - 1, img['height'] - 1], dtype=np.float32)

                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {"img_path": img_path,
                        "img_shape": img_shape,
                        "joints_coord_cam": joints_coord_cam,
                        "joints_coord_img": joints_coord_img,
                        "root_joint_cam": root_joint_cam,
                        "bbox": bbox,
                        "cam_param": cam_param,
                        "image_id": image_id,
                        "mano_pose": mano_pose,
                        "mano_shape": mano_shape,
                        'hand_type': hand_type}

            self.dataset.append(data)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data .
        """
        data = self.dataset[idx].copy()
        img_path, img_shape, bbox = (
            data["img_path"],
            data["img_shape"],
            data["bbox"],
        )

        hand_type = data['hand_type']
        do_flip = (hand_type == 'left')

        # load image
        image = load_img(img_path)

        # transform
        image, img2bb_trans, bb2img_trans, rot, scale = augmentation(image, bbox, self.data_split)
        image = torch.FloatTensor(image)

        if self.data_split == 'train':
            ## 2D joint coordinate
            joints_img = data['joints_coord_img']
            if do_flip:
                joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize to [0,1]
            joints_img[:, 0] /= 256
            joints_img[:, 1] /= 256

            ## 3D joint camera coordinate
            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative
            if do_flip:
                joints_coord_cam[:, 0] *= -1

            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

            ## mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']

            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1, 3)
            if do_flip:
                mano_pose[:, 1:] *= -1
            root_pose = mano_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)
            # print(joints_coord_cam)



            inputs = image.reshape((3, 256, 256))
            targets = {'joints_img': joints_img, 'joints_coord_cam': joints_coord_cam, 'mano_pose': mano_pose,
                       'mano_shape': mano_shape}
            meta_info = {'root_joint_cam': root_joint_cam}

        else:
            joints_img = data['joints_coord_img']


            # normalize to [0,1]
            joints_img[:, 0] /= 256
            joints_img[:, 1] /= 256

            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            ## mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']

            inputs = image.reshape((3, 256, 256))
            targets = {'joints_img': joints_img, 'joints_coord_cam': joints_coord_cam, 'mano_pose': mano_pose,
                       'mano_shape': mano_shape}
            meta_info = {'root_joint_cam': root_joint_cam}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.dataset
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]

            out = outs[n]

            joints_out = out['joints_coord_cam']

            # root centered
            joints_out -= joints_out[self.root_joint_idx]

            # flip back to left hand
            if annot['hand_type'] == 'left':
                joints_out[:, 0] *= -1

            # root align
            gt_root_joint_cam = annot['root_joint_cam']
            joints_out += gt_root_joint_cam

            # GT and rigid align
            joints_gt = annot['joints_coord_cam']
            joints_out_aligned = rigid_align(joints_out, joints_gt)

            # m to mm
            joints_out *= 1000
            joints_out_aligned *= 1000
            joints_gt *= 1000
            self.eval_results[0].append(np.sqrt(np.sum((joints_out - joints_gt) ** 2, 1)).mean())
            self.eval_results[1].append(np.sqrt(np.sum((joints_out_aligned - joints_gt) ** 2, 1)).mean())

    def eval_result(self):
        MPJPE = self.eval_results[0][-1]
        PA_MPJPE = self.eval_results[1][-1]
        return MPJPE, PA_MPJPE

    def mean_eval_result(self):
        MPJPE = np.mean(self.eval_results[0])
        PA_MPJPE = np.mean(self.eval_results[1])
        return MPJPE, PA_MPJPE



if __name__ == '__main__':
    data = DEX_YCBVDataset('/home/ana/Study/CVPR/lego/dataset', 'train')
    img, tg, info = data.__getitem__(0)