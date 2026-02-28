# -*- encoding: utf-8 -*-
'''
@File    :   FPdataset.py
@Time    :   2026/02/28 18:31:14
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2026, Zhiyu Pan, Tsinghua University. All rights reserved
'''
import os
import os.path as osp
import numpy as np

from PIL import Image
from scipy import ndimage as sndi 

import cv2
from torch.utils.data import Dataset
cv2.setUseOptimized(True)

def mkdir(path):  
    if not os.path.exists(path):
        os.makedirs(path)

def affine_matrix(scale=1.0, theta=0.0, trans=np.zeros(2), trans_2=np.zeros(2)):  # apply on the {?} coordinate
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale 
    t = np.dot(R, trans) + trans_2
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])

def coarse_center(img, img_ppi=500):
    img = np.rint(img).astype(np.uint8)
    ksize1 = int(19 * img_ppi / 500)
    ksize2 = int(5 * img_ppi / 500)
    seg = cv2.GaussianBlur(img, ksize=(ksize1, ksize1), sigmaX=0, borderType=cv2.BORDER_REPLICATE)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2, ksize2)))
    seg = seg.astype(np.float32)

    grid = np.stack(np.meshgrid(*[np.arange(x) for x in img.shape[:2]], indexing="ij")).reshape(2, -1)
    img_c = (seg.reshape(1, -1) * grid).sum(1) / seg.sum().clip(1e-6, None)
    return img_c
    
class FingerPoseEvalDataset(Dataset):
    def __init__(
        self,
        folder_path, # dataset_name
        img_ppi=500,
        seg_zoom=4,
        middle_shape=(512, 512),
        save_folder='votingpose',
        seed=None,
    ) -> None:
        super().__init__()
        self.folder_path = folder_path
        # self.phase = phase
        self.img_ppi = img_ppi
        self.seg_zoom = seg_zoom
        self.middle_shape = np.array(middle_shape)
        self.seed = seed

        self.scale = self.img_ppi * 1.0 / 500
        ppad = 128 if self.img_ppi == 30 else 64 # padding for different ppi
        self.tar_shape = np.rint(np.maximum(np.ones(2), (self.middle_shape * self.scale + ppad / 2) // ppad) * ppad).astype(int)

        self.items = os.listdir(folder_path)
        # replace the image folder with the pose folder
        self.save_folder = folder_path.replace("image", save_folder)
        mkdir(self.save_folder)

    def load_img(self, img_path):
        img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
        return img

    def padding_img(self, img, tar_shape, cval=0):
        src_shape = np.array(img.shape[:2])
        padding = np.maximum(src_shape, tar_shape) - src_shape
        img = np.pad(img, ((0, padding[0]), (0, padding[1])), constant_values=cval)
        return img[: tar_shape[0], : tar_shape[1]]

    def resize_img(self, img, ratio, order=1, cval=0):
        resized_img = sndi.zoom(img, ratio, order=order, cval=cval)
        return resized_img

    def __len__(self):
        return len(self.items)
    
    def process_img(self, img, img_c=None):
        if self.scale < 1:
            img = sndi.uniform_filter(img, min(15, np.rint(1.2 / self.scale)))
            img = sndi.zoom(img, self.scale, order=1)

        center = self.tar_shape[::-1] / 2.0
        img_c = np.array(img.shape[1::-1]) / 2.0 if img_c is None else img_c

        T = affine_matrix(scale=1, theta=0, trans=-img_c, trans_2=center)
        img = cv2.warpAffine(img, T[:2], dsize=tuple(self.tar_shape[::-1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return img, T

    def __getitem__(self, index):
        item = self.items[index]

        img = self.load_img(os.path.join(self.folder_path, item)) # + img_format 'img', 'pose_2d', "seg"
        img, T = self.process_img(img)

        # the name of output result
        path = os.path.join(self.save_folder, item)
        path = path[:path.rfind(".")] + ".txt"
        return {
            # "img_ori": img_ori[None].astype(np.float32),
            "img": img[None].astype(np.float32),
            "T": T.astype(np.float32),
            "name": path,
        }


class Descdataset(Dataset):
    def __init__(
        self, folder_path, method_name, tar_shape=(299, 299), middle_shape=(448, 448), is_stn=False, pose_name='',
    ) -> None:
        super().__init__()
        self.is_stn = is_stn
        self.folder_path = folder_path
        self.path_lst = os.listdir(folder_path)
        # if using_gtpose:
        if len(pose_name) == 0 or is_stn: # if stn, do not use pose
            self.posefolder_path = ''
        else:
            self.posefolder_path = folder_path.replace("image", pose_name)
        # judge posegt folder exists
        if not os.path.exists(self.posefolder_path):
            self.posefolder_path = ''
            # raise ValueError(f"posegt folder {self.posefolder_path} does not exist")
            print(f"Do not use the pose")
        else:
            print(f"Using the pose from {self.posefolder_path}")
        # else:
        #     self.posefolder_path = folder_path.replace("image", "pose")
        self.mask_folder = folder_path.replace("image", "fingernet/seg")
        self.tar_shape = np.array(tar_shape)
        self.middle_shape = np.array(middle_shape)
        self.desc_folder = folder_path.replace("image", f"{method_name}_feat_{pose_name}")
        mkdir(self.desc_folder)


    def __len__(self):
        return len(self.path_lst)

    def process_img(self, img_ori, pose_2d=None, mask=None):
        center = self.tar_shape[::-1] / 2.0
        shift = np.zeros(2)
        if pose_2d is not None:
            img_c = pose_2d[:2]
            theta = pose_2d[2]
        else:
            img_c = coarse_center(img_ori, img_ppi=500)[::-1]
            # img_c = np.array(img_ori.shape[:2]) / 2.0
            theta = 0

        T = affine_matrix(
            scale=self.tar_shape[0] * 1.0 / self.middle_shape[0],
            theta=np.deg2rad(theta),
            trans=-img_c,
            trans_2=center + shift,
        )

        img = cv2.warpAffine(
            (img_ori.astype(np.float32) - 127.5) / 127.5,
            T[:2],
            dsize=tuple(self.tar_shape[::-1]),
            flags=cv2.INTER_LINEAR,
        )

        if mask is not None:
            mask = cv2.warpAffine(
                mask.astype(np.float32) / 255.0,
                T[:2],
                dsize=tuple(self.tar_shape[::-1]),
                flags=cv2.INTER_LINEAR,
            )
            # resize the mask 
            # mask = cv2.resize(mask, (self.tar_shape[::-1]//16), interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            return img, T,  mask
        else:
            return img, T

    def __getitem__(self, index):
        item = self.path_lst[index]
        mask = None
        img_ori = np.asarray(
            cv2.imread(osp.join(self.folder_path, item), cv2.IMREAD_GRAYSCALE),
            dtype=np.float32,
        )
        # 判断img_ori是否为None
        img_shape = np.array(img_ori.shape)
        pose_file = osp.join(self.posefolder_path, item[: item.rfind(".")] + ".txt")
        
        if osp.exists(pose_file) and not self.is_stn:
            pose_2d = np.loadtxt(pose_file)
        else:
            pose_2d = None

        if osp.exists(osp.join(self.mask_folder, item.split('.')[0] + '.bmp')): # 
            mask = np.asarray(
                cv2.imread(osp.join(self.mask_folder, item.split('.')[0] + '.bmp'), cv2.IMREAD_GRAYSCALE),
                dtype=np.float32,
            )
            img, T,  mask = self.process_img(img_ori, pose_2d, mask)

        else:
            img, T = self.process_img(img_ori, pose_2d)

        # save the descriptor in pickle form
        save_path = osp.join(self.desc_folder, item[: item.rfind(".")] + ".pkl")

        if mask is not None:
            # 验证合法性
            return {
                "img": img[None].astype(np.float32),
                "img_shape": img_shape,
                "mask": mask[None].astype(np.float32),
                "T": T.astype(np.float32),
                # "pose_2d": pose_2d,
                "name": save_path,
            }
        else:
            return {"img": img[None].astype(np.float32), 
                    "img_shape": img_shape,
                    # "pose_2d": pose_2d,
                    "T": T.astype(np.float32),
                    "name": save_path}