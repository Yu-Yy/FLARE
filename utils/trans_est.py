'''
Description: 
Author: Xiongjun Guan
Date: 2024-06-13 15:27:51
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-07-22 14:34:49

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def selectMax(x):
    x = x / torch.max(x, dim=1, keepdim=True).values.clamp_min(1e-8)
    x = torch.where(x > 0.999, x, torch.zeros_like(x))
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return x


def classify2vector_trans(pred_xy, out_form, trans_num_classes, eps=1e-6):
    trans_const = 256
    if out_form == "claSum":
        trans_tensor = np.linspace(-trans_const, trans_const,
                                   trans_num_classes // 2)
        trans_tensor = torch.FloatTensor(trans_tensor).to(pred_xy.device)
        _, c = pred_xy.shape
        x_pred = pred_xy[:, :c // 2]
        y_pred = pred_xy[:, c // 2:]
        x_pred = (torch.sum(x_pred * trans_tensor, dim=-1) /
                  (torch.sum(x_pred, dim=-1) + eps))
        y_pred = (torch.sum(y_pred * trans_tensor, dim=-1) /
                  (torch.sum(y_pred, dim=-1) + eps))
    elif out_form == "claMax":
        trans_tensor = np.linspace(-trans_const, trans_const,
                                   trans_num_classes // 2)
        trans_tensor = torch.FloatTensor(trans_tensor).to(pred_xy.device)
        _, c = pred_xy.shape
        x_pred = pred_xy[:, :c // 2]
        y_pred = pred_xy[:, c // 2:]
        x_pred = selectMax(x_pred)
        y_pred = selectMax(y_pred)
        x_pred = (torch.sum(x_pred * trans_tensor, dim=-1) /
                  (torch.sum(x_pred, dim=-1) + eps))
        y_pred = (torch.sum(y_pred * trans_tensor, dim=-1) /
                  (torch.sum(y_pred, dim=-1) + eps))
    elif out_form == "reg":
        x_pred = pred_xy[:, 0]
        y_pred = pred_xy[:, 1]

    elif out_form == "heat":
        trans_tensor = np.linspace(-trans_const, trans_const, 512 // 16)
        X_grid, Y_grid = np.meshgrid(trans_tensor, trans_tensor)
        X_grid = torch.FloatTensor(X_grid).to(pred_xy.device)[None, None, :, :]
        Y_grid = torch.FloatTensor(Y_grid).to(pred_xy.device)[None, None, :, :]
        x_pred = (torch.sum(pred_xy * X_grid, dim=(2, 3)) /
                  (torch.sum(pred_xy, dim=(2, 3)) + eps))
        y_pred = (torch.sum(pred_xy * Y_grid, dim=(2, 3)) /
                  (torch.sum(pred_xy, dim=(2, 3)) + eps))

    if len(x_pred.shape) > 1:
        x_pred = x_pred.squeeze(1)
        y_pred = y_pred.squeeze(1)
    vec_xy = torch.stack([x_pred, y_pred]).transpose(1, 0)

    return vec_xy


def classify2vector_rot(pred_theta, out_form, rot_num_classes, eps=1e-6):

    if out_form == "claSum":
        rot_tensor = np.linspace(-np.pi, np.pi, rot_num_classes)
        rot_tensor = torch.FloatTensor(rot_tensor).to(pred_theta.device)
        cos_pred = (torch.sum(pred_theta * torch.cos(rot_tensor), dim=-1) /
                    (torch.sum(pred_theta, dim=-1) + eps))
        sin_pred = (torch.sum(pred_theta * torch.sin(rot_tensor), dim=-1) /
                    (torch.sum(pred_theta, dim=-1) + eps))

    elif out_form == "claMax":
        rot_tensor = np.linspace(-np.pi, np.pi, rot_num_classes)
        rot_tensor = torch.FloatTensor(rot_tensor).to(pred_theta.device)
        pred_theta = selectMax(pred_theta)
        cos_pred = (torch.sum(pred_theta * torch.cos(rot_tensor), dim=-1) /
                    (torch.sum(pred_theta, dim=-1) + eps))
        sin_pred = (torch.sum(pred_theta * torch.sin(rot_tensor), dim=-1) /
                    (torch.sum(pred_theta, dim=-1) + eps))

    elif out_form == "reg_ang":
        ang_pred = pred_theta
        rad_pred = torch.deg2rad(pred_theta)
        cos_pred = torch.cos(rad_pred)
        sin_pred = torch.sin(rad_pred)

    elif out_form == "reg_tan":
        pred_b1 = pred_theta[:, 0]
        pred_b2 = pred_theta[:, 1]
        norm_pred = torch.sqrt(torch.square(pred_b1) + torch.square(pred_b2))
        cos_pred = torch.div(pred_b1, norm_pred)
        sin_pred = torch.div(pred_b2, norm_pred)

    if len(cos_pred.shape) > 1:
        cos_pred = cos_pred.squeeze(1)
        sin_pred = sin_pred.squeeze(1)

    ang_pred = torch.rad2deg(torch.arctan2(sin_pred, cos_pred))

    if len(ang_pred.shape) > 1:
        ang_pred = ang_pred.squeeze(1)

    vec_theta = torch.stack([cos_pred, sin_pred, ang_pred]).transpose(1, 0)

    return vec_theta
