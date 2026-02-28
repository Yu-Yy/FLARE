# -*- encoding: utf-8 -*-
'''
@File    :   extract_FDD.py
@Time    :   2026/02/28 18:08:19
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2026, Zhiyu Pan, Tsinghua University. All rights reserved
'''

import argparse
from tqdm import tqdm
import os
import numpy as np
import torch
import yaml
from utils.misc import load_model
import sys
from datasets import FPdataset as datasets
from models.model_zoo import FDD
import torch.backends.cudnn as cudnn
import logging
from easydict import EasyDict as edict
from torch.utils.data import ConcatDataset
import pickle
import pandas as pd
import time

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extracting(config):
    # learn the 
    dataset_folder = config.folder
    logging.info(f"Create the datasets")
    search_folder = os.path.join(dataset_folder, 'image', 'query')
    gallery_folder = os.path.join(dataset_folder, 'image', 'gallery')
    e_datasets = []
    
    for dataset_ in [search_folder, gallery_folder]:
        valid_dataset = datasets.Descdataset(
            dataset_, 
            config.NAME, # method name
            tar_shape=config.MODEL.tar_shape,
            middle_shape=config.MODEL.middle_shape,
            pose_name=config.pose, # using the gt_pose
        )
        e_datasets.append(valid_dataset)

    desc_dataset = ConcatDataset(e_datasets)
    desc_dataloader = torch.utils.data.DataLoader(desc_dataset,
                    batch_size=config.DATASET.BATCH_SIZE, shuffle=False,
                    num_workers=config.DATASET.NUM_WORKERS, pin_memory=True)
    
    logging.info(f"Create the model about descriptor")
    desc_model = FDD(
        ndim_feat=config.MODEL.ndim_feat,
        input_norm = config.MODEL.input_norm,
        tar_shape=config.MODEL.tar_shape,)
    
    desc_model = desc_model.cuda()
    with torch.no_grad():
        desc_model = torch.nn.DataParallel(desc_model)
    model_path = f'model_weights/desc_model.pth.tar' # 
    load_model(desc_model, model_path) # 
    logging.info(f"Start to estimate the descriptor of the dataset") 
    valid_desc(desc_dataloader, desc_model)

def valid_desc(dataloader, model):
    model.eval()
    total_time = 0
    with torch.no_grad():
        with tqdm(total = len(dataloader.dataset), desc=f"EVAl") as pbar:
            for iterx, item in enumerate(dataloader):
                img = item["img"].cuda()
                if "mask" in item.keys():
                    gt_masks = item["mask"].flatten(1)
                    gt_masks = gt_masks.numpy()
                names = item["name"]
                # calculate the time
                start = time.time()
                outputs = model.module.get_embedding(img)
                total_time += time.time() - start
                features = outputs["feature"].cpu().numpy()
                masks = outputs["mask"].cpu().numpy()
                B = img.shape[0]
                pbar.update(B)
                for b in range(B):
                    save_path  = names[b]
                    save_dict = {}
                    save_dict["feature"] = features[b]
                    save_dict["mask"] = masks[b]
                    with open(save_path, 'wb') as f:
                        pickle.dump(save_dict, f)    
    logging.info(f"Descriptor estimation finished") 
    logging.info(f"Average time for each image is {total_time/len(dataloader.dataset):.2e}s")

def calculate_score(feat1, feat2, mask1, mask2, ndim_feat=12, binary=False, verbose=False):
    '''
    The function to calculate the score between two images or two set images
    '''
    feat1_dense = feat1
    feat1_mask = np.tile(mask1, (1, ndim_feat))
    feat2_dense = feat2
    feat2_mask = np.tile(mask2, (1, ndim_feat))
    if verbose:
        end = time.time()
    if binary:
        # THRESHS = {0: 0.2, 1: 0.002, 2: 0.5} # 0 for plain, 1 for rolled, 2 for latent
        feat1_dense = (feat1_dense > 0).astype(np.float32)
        feat2_dense = (feat2_dense > 0).astype(np.float32)
        feat1_mask = (feat1_mask > 0.5).astype(np.float32)
        feat2_mask = (feat2_mask > 0.2).astype(np.float32)

        n12 = np.matmul(feat1_mask, feat2_mask.T)
        d12 = (n12 - np.matmul((feat1_mask * feat1_dense), (feat2_mask * feat2_dense).T) - np.matmul((feat1_mask * (1 - feat1_dense)), (feat2_mask * (1 - feat2_dense)).T))
        score = 1 - 2 * np.where(n12 > 0, d12 / n12.clip(1e-3, None), 0.5)

    else:
        x1 = np.sqrt(np.matmul(feat1_mask * feat1_dense**2, feat2_mask.T))
        x2 = np.sqrt(np.matmul(feat1_mask, (feat2_dense**2 * feat2_mask).T))
        x12 = np.matmul(feat1_mask * feat1_dense, (feat2_mask * feat2_dense).T)
        score = x12 / (x1 * x2).clip(1e-3, None)

    if verbose:
        total = time.time() - end
        num = score.size
        print(f"matching consumes: {total}s, speed: {(total / num):.2e}/pair")

    return score

# FDD's Matching
def matching(config):
    folder = config.folder
    feat_folder = os.path.join(folder, f'{config.NAME}_feat_{config.pose}') # if not config.pose else os.path.join(folder, f'{config.NAME}_feat_gtpose')
    search_folder = os.path.join(feat_folder, 'query')
    gallery_folder = os.path.join(feat_folder, 'gallery')
    score_file = os.path.join(feat_folder, f'score_{config.NAME}.csv') if not config.binary else os.path.join(feat_folder,f'score_binary_{config.NAME}.csv') 
    search_files = os.listdir(search_folder)
    search_files.sort()
    gallery_files = os.listdir(gallery_folder)
    gallery_files.sort()

    search_ = []
    gallery_ = []
    for i, search_file in tqdm(enumerate(search_files)):
        file_path = os.path.join(search_folder, search_file)
        with open(file_path, 'rb') as f:
            search_.append(pickle.load(f))
    for j, gallery_file in enumerate(gallery_files):
        file_path = os.path.join(gallery_folder, gallery_file)
        with open(file_path, 'rb') as f:
            gallery_.append(pickle.load(f))
    search_feat = np.concatenate([search_[i]['feature'][None] for i in range(len(search_))], axis=0)
    gallery_feat = np.concatenate([gallery_[i]['feature'][None] for i in range(len(gallery_))], axis=0)

    search_mask = np.concatenate([search_[i]['mask'][None] for i in range(len(search_))], axis=0)
    gallery_mask = np.concatenate([gallery_[i]['mask'][None] for i in range(len(gallery_))], axis=0)
    
    score_matrix = calculate_score(search_feat, gallery_feat, search_mask, gallery_mask, config.MODEL.ndim_feat * 2, config.binary, verbose=True)
    score_df = pd.DataFrame(score_matrix)
    score_df.columns = gallery_files
    score_df.index = search_files
    score_df.to_csv(score_file) # write the score matrix as a csv file
    logging.info(f"Score matrix saved to {score_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FDD evaluation')
    parser.add_argument("--folder", "-f", required=True ,type=str) # 
    parser.add_argument("--gpu", "-g", type=str, default='0', help="the gpu id")
    parser.add_argument("--binary", "-b", action='store_true', help="binary score")
    parser.add_argument("--pose", "-p", type=str, default='pose', help="the pose file")
    args = parser.parse_args()
    CUDA_VISIBLE_DEVICES= args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    yaml_path = 'model_weights/desc_configs.yaml' # control the used model
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    params = edict(yaml.safe_load(open(yaml_path, 'r')))
    # update the args into the params
    params.update(vars(args))
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    extracting(params)
    matching(params)