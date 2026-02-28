# -*- encoding: utf-8 -*-
'''
@File    :   extract_RegressionPose.py
@Time    :   2026/02/28 18:08:33
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2026, Zhiyu Pan, Tsinghua University. All rights reserved
'''
import numpy as np
import torch
import cv2
from models.model_zoo import FingerPose_2D_Single
import argparse
import os
from torch.utils.data import ConcatDataset
import logging
# from datasets import data_loader
from datasets import FPdataset as datasets
import torch.backends.cudnn as cudnn
import time
from tqdm import tqdm
from utils.trans_est import classify2vector_rot, classify2vector_trans


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model, ckp_path, by_name=False):
    def remove_module_string(k):
        items = k.split(".")
        items = items[0:1] + items[2:]
        return ".".join(items)

    if isinstance(ckp_path, str):
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        if "model" in ckp.keys():
            ckp_model_dict = ckp["model"]
        else:
            ckp_model_dict = ckp
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {remove_module_string(k): v for k, v in ckp_model_dict.items()}
    if by_name:
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in ckp_model_dict.items() if k in model_dict and v.shape == model_dict[k].shape} 
        model_dict.update(state_dict)
        ckp_model_dict = model_dict

    if hasattr(model, "module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)

def main(config):
    dataset_folder = config.folder # get the folder
    logging.info(f"Create the datasets about pose")
    search_folder = os.path.join(dataset_folder, 'image','query')
    gallery_folder = os.path.join(dataset_folder, 'image','gallery')
    # set the dataset
    e_datasets = []
    for dataset_ in [search_folder, gallery_folder]:
        valid_dataset = datasets.FingerPoseEvalDataset(
            dataset_, 
            img_ppi=500, 
            seg_zoom=4, 
            middle_shape=(512, 512),
            save_folder='RegressionPose', # rot 180 写成90了
        )
        e_datasets.append(valid_dataset)
    pose_dataset = ConcatDataset(e_datasets)
    pose_dataloader = torch.utils.data.DataLoader(pose_dataset,
                    batch_size=32, shuffle=False,
                    num_workers=16, pin_memory=True)
    logging.info(f"Create the model about pose")
    
    model = FingerPose_2D_Single(
        inp_mode='fp', 
        trans_out_form='claSum',
        trans_num_classes=512,
        rot_out_form='claSum',
        rot_num_classes=180,)

    model = model.cuda()
    with torch.no_grad():
        model = torch.nn.DataParallel(model) # TODO: single GPU
    model_path = 'model_weights/RegressionPose.pth'
    load_model(model, model_path)
    logging.info(f"Start to estimate the pose of the dataset")
    valid_pose(pose_dataloader, model)

def valid_pose(dataloader, model):
    model.eval()
    total_time = 0
    with torch.no_grad():
        with tqdm(total = len(dataloader.dataset), desc=f"EVAl") as pbar:
            for iterx, item in enumerate(dataloader):
                img = item["img"].cuda()
                names = item["name"]
                T_batch = item["T"]
                start = time.time()
                [pred_xy, pred_theta] = model(img)
                vec_xy = classify2vector_trans(pred_xy,
                                           out_form='claSum',
                                           trans_num_classes=512)
                vec_xy = vec_xy + 256 # to the middle of the image
                vec_theta = classify2vector_rot(pred_theta,
                                            out_form='claSum',
                                            rot_num_classes=180)
                pred_pose_2d = torch.cat([vec_xy, vec_theta[:,2:3]], dim=1).cpu().numpy()
                total_time += time.time() - start
                B = img.shape[0]
                pbar.update(B)
                for b in range(B):
                    name = names[b]
                    pose_2d = pred_pose_2d[b]
                    T = T_batch[b].numpy()
                    T_inv = np.linalg.inv(T)
                    pose_2d[:2] = np.dot(T_inv[:2, :2], pose_2d[:2]) + T_inv[:2, 2]
                    pose_2d[2] = (pose_2d[2] + 180) % 360 - 180 
                    # save the pose_2d
                    mkdir(os.path.dirname(name))
                    np.savetxt(name, pose_2d)
    logging.info(f"Pose estimation finished for {len(dataloader.dataset)} images")
    logging.info(f"Average time for each image is {total_time/len(dataloader.dataset):.2e}s/sample")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Estimating the finger pose (Regress) of the dataset")
    parser.add_argument("--folder", "-f", required=True, type=str, help="the folder of the dataset")
    parser.add_argument("--gpu", "-g", type=str, default='0', help="the gpu id")
    args = parser.parse_args() 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    main(args)