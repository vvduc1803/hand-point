import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO
from main.config import cfg
from common.utils.preprocessing import load_img, load_depth_img, get_bbox, process_bbox, generate_patch_image, \
    augmentation, augmentation_depth
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from common.utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton, render_mesh, \
    vis_3d_skeleton
from common.utils.mano import MANO

mano = MANO()


class DEX_YCB(torch.utils.data.Dataset):
    def __init__(self, transform, data_split, cam_scale=256, num_pt=1024):
        self.transform = transform
        self.data_split = data_split if data_split == 'train' else 'test'
        self.cam_scale = cam_scale
        self.num_pt = num_pt

        self.root_dir = osp.join('/home/ana/Study/CVPR/HandOccNet', 'data', 'DEX_YCB', 'data')
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0

        self.xmap = np.array([[j for i in range(256)] for j in range(256)])
        self.ymap = np.array([[i for i in range(256)] for j in range(256)])

        self.datalist = self.load_data()
        if self.data_split != 'train':
            self.eval_result = [[], []]  # [mpjpe_list, pa-mpjpe_list]

    def load_data(self):
        db = COCO(osp.join(self.annot_path, "DEX_YCB_s0_{}_subset_data.json".format(self.data_split)))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, img['file_name'])
            depth_path = img_path.replace("color_", "aligned_depth_to_color_").replace(".jpg", ".png")
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
                        "depth_path": depth_path,
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

                data = {"img_path": img_path,
                        "depth_path": depth_path,
                        "img_shape": img_shape,
                        "joints_coord_cam": joints_coord_cam,
                        "root_joint_cam": root_joint_cam,
                        "bbox": bbox,
                        "cam_param": cam_param,
                        "image_id": image_id,
                        'hand_type': hand_type}

            datalist.append(data)
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, depth_path, img_shape, bbox = (
            data["img_path"],
            data["depth_path"],
            data["img_shape"],
            data["bbox"],
        )

        hand_type = data['hand_type']
        do_flip = (hand_type == 'left')

        # img
        img = load_img(img_path)
        depth = load_depth_img(depth_path).reshape((480, 640, 1))

        # orig_img = copy.deepcopy(img)[:,:,::-1]
        img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, do_flip=do_flip)
        depth = augmentation_depth(depth, bbox, self.data_split, do_flip=do_flip)

        img = self.transform(img.astype(np.float32)) / 255.
        depth = self.transform(depth.astype(np.float32)) / 255.

        mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))

        choose = mask_depth.flatten().nonzero()[0]

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth.numpy().flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        fx, fy = data['cam_param']['focal']
        cx, cy = data['cam_param']['princpt']

        cam_scale = self.cam_scale
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cx) * pt2 / fx
        pt1 = (xmap_masked - cy) * pt2 / fy
        cloud = torch.permute(torch.tensor(np.concatenate((pt0, pt1, pt2), axis=1)), (1, 0))

        if self.data_split == 'train':
            ## 2D joint coordinate
            joints_img = data['joints_coord_img']
            if do_flip:
                joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            # normalize to [0,1]
            joints_img[:, 0] /= cfg.input_img_shape[1]
            joints_img[:, 1] /= cfg.input_img_shape[0]

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

            inputs = {'img': img, 'cloud': cloud}
            targets = {'joints_img': joints_img, 'joints_coord_cam': joints_coord_cam, 'mano_pose': mano_pose,
                       'mano_shape': mano_shape}
            meta_info = {'root_joint_cam': root_joint_cam}

        else:
            root_joint_cam = data['root_joint_cam']
            inputs = {'img': img, 'cloud': cloud}
            targets = {}
            meta_info = {'root_joint_cam': root_joint_cam}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
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

            self.eval_result[0].append(np.sqrt(np.sum((joints_out - joints_gt) ** 2, 1)).mean())
            self.eval_result[1].append(np.sqrt(np.sum((joints_out_aligned - joints_gt) ** 2, 1)).mean())

    def print_eval_result(self, test_epoch):
        # print("Epoch %d evaluation result:" % test_epoch)
        MPJPE = np.mean(self.eval_result[0])
        PA_MPJPE = np.mean(self.eval_result[1])
        # print("MPJPE : %.2f mm" % MPJPE)
        # print("PA MPJPE : %.2f mm" % PA_MPJPE)
        return MPJPE, PA_MPJPE
