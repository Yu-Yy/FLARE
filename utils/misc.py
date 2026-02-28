"""
This file (functions.py) is designed for:
    functions
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
from glob import glob
import numpy as np
import time
import torch
from scipy import ndimage as ndi, integrate


def intensity_normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min).clamp_min(1e-6)


# def save_model(model, optim, schedule, epoch, path, save_ckp_num=5):
#     if hasattr(model, "module"):
#         model_state = model.module.state_dict()
#     else:
#         model_state = model.state_dict()
#     torch.save({"model": model_state, "optim": optim.state_dict(), "schedule": schedule.state_dict(), "epoch": epoch}, path)

#     # remove redundent checkpoint
#     name_lst = glob(osp.join(osp.dirname(path), "e_*.pth.tar"))
#     name_lst = [osp.basename(x) for x in name_lst]
#     name_lst.sort()
#     if len(name_lst) > save_ckp_num:
#         os.remove(osp.join(osp.dirname(path), name_lst[0]))
def save_model(model, optimizer, Epoch, global_step, path, best=False, lr_scheduler=None):
    if hasattr(model, "module"):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    state = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "epoch": Epoch,
        "global_step": global_step,
    }
    if best:
        torch.save(state, path + "/best_model.pth")
    else:
        torch.save(state, path + "/model_{}.pth".format(Epoch))
    # if exit more than 5 models, delete the oldest one
    m_files = os.listdir(path)
    m_files = [f for f in m_files if f.endswith(".pth") and "best" not in f]
    # sort the files according to the epoch
    m_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    if len(m_files) > 2:
        os.remove(os.path.join(path, m_files[0]))

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


def generate_random_seed():
    seed = hash(time.time()) % 10000
    return seed


def my_load_state_dict(model, ckp_path):
    ckp = torch.load(ckp_path, lambda storage, _: storage)
    ckp_state_dict = ckp["model"]
    model_state_dict = model.state_dict()
    to_be_updated = {
        k: v for k, v in ckp_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    model_state_dict.update(to_be_updated)
    print("length of pretained weights: ", len(list(to_be_updated.keys())))
    model.load_state_dict(model_state_dict)


def surface_integrate(normal_int, mask_int):
    grad_x = np.stack((normal_int[2], normal_int[0]))
    grad_y = np.stack((normal_int[2], -normal_int[1]))
    # grad_x = grad_x / np.linalg.norm(grad_x, axis=0, keepdims=True).clip(1e-6, None)
    # grad_y = grad_y / np.linalg.norm(grad_y, axis=0, keepdims=True).clip(1e-6, None)

    x_int, y_int, z_int, mask_min = integrate_3d_gradient(grad_x[0], grad_y[0], grad_x[1], grad_y[1], mask_int)

    # z_int[mask_min > 0] = z_int[mask_min > 0] - z_int[mask_min > 0].min() + 1
    # z_int[~mask_min] = 0

    return np.stack((x_int, y_int, z_int))


def integrate_3d_gradient(grad_h, grad_w, grad_zh, grad_zw, mask):
    h, w = mask.shape

    # mask_erode = ndi.morphology.binary_erosion(mask, np.ones([10, 10]))
    # mag = np.sqrt(grad_zh ** 2 + grad_zw ** 2)
    # mag[~mask_erode] = mag.max() + 1
    # min_h, min_w = np.unravel_index(mag.argmin(), mag.shape)

    mag = ndi.distance_transform_edt(mask)
    min_h, min_w = np.unravel_index(mag.argmax(), mag.shape)

    coord_h, coord_w, coord_z, mask_min = integrate_trapz(min_h, min_w, grad_h, grad_w, grad_zh, grad_zw, mask)

    return coord_h, coord_w, coord_z, mask_min


def integrate_trapz(h_c, w_c, grad_h, grad_w, grad_zh, grad_zw, mask):
    # h coordinate
    rows_u = -grad_h[h_c::-1]
    rows_u_int = integrate.cumtrapz(rows_u, axis=0, initial=0)[::-1]
    rows_d = grad_h[h_c:]
    rows_d_int = integrate.cumtrapz(rows_d, axis=0, initial=0)
    coord_h = np.concatenate((rows_u_int, rows_d_int[1:]), axis=0)

    # w coordinate
    cols_l = -grad_w[:, w_c::-1]
    cols_l_int = integrate.cumtrapz(cols_l, axis=1, initial=0)[:, ::-1]
    cols_r = grad_w[:, w_c:]
    cols_r_int = integrate.cumtrapz(cols_r, axis=1, initial=0)
    coord_w = np.concatenate((cols_l_int, cols_r_int[:, 1:]), axis=1)

    # y-x
    col0_z_u = -grad_zh[h_c::-1, w_c]
    col0_h_u = -coord_h[h_c::-1, w_c]
    # col0_u_int = integrate.cumtrapz(col0_z_u, x=col0_h_u, initial=0)[::-1]
    col0_u_int = integrate.cumtrapz(col0_z_u, initial=0)[::-1]
    col0_z_d = grad_zh[h_c:, w_c]
    col0_h_d = coord_h[h_c:, w_c]
    # col0_d_int = integrate.cumtrapz(col0_z_d, x=col0_h_d, initial=0)
    col0_d_int = integrate.cumtrapz(col0_z_d, initial=0)
    col0_int = np.concatenate((col0_u_int, col0_d_int[1:]))
    col0_int[mask[:, w_c] == 0] = np.nan

    cols_z_l = -grad_zw[:, w_c::-1]
    cols_w_l = -coord_w[:, w_c::-1]
    # cols_l_int = integrate.cumtrapz(cols_z_l, x=cols_w_l, axis=1, initial=0)[:, ::-1]
    cols_l_int = integrate.cumtrapz(cols_z_l, axis=1, initial=0)[:, ::-1]
    cols_z_r = grad_zw[:, w_c:]
    cols_w_r = coord_w[:, w_c:]
    # cols_r_int = integrate.cumtrapz(cols_z_r, x=cols_w_r, axis=1, initial=0)
    cols_r_int = integrate.cumtrapz(cols_z_r, axis=1, initial=0)
    cols_int = np.concatenate((cols_l_int, cols_r_int[:, 1:]), axis=1)
    yx_int = cols_int + col0_int[:, None]

    # x-y
    row0_z_l = -grad_zw[h_c, w_c::-1]
    row0_w_l = -coord_w[h_c, w_c::-1]
    # row0_l_int = integrate.cumtrapz(row0_z_l, x=row0_w_l, initial=0)[::-1]
    row0_l_int = integrate.cumtrapz(row0_z_l, initial=0)[::-1]
    row0_z_r = grad_zw[h_c, w_c:]
    row0_w_r = coord_w[h_c, w_c:]
    # row0_r_int = integrate.cumtrapz(row0_z_r, x=row0_w_r, initial=0)
    row0_r_int = integrate.cumtrapz(row0_z_r, initial=0)
    row0_int = np.concatenate((row0_l_int, row0_r_int[1:]))
    row0_int[mask[h_c] == 0] = np.nan

    rows_z_u = -grad_zh[h_c::-1]
    rows_h_u = -coord_h[h_c::-1]
    # rows_u_int = integrate.cumtrapz(rows_z_u, x=rows_h_u, axis=0, initial=0)[::-1]
    rows_u_int = integrate.cumtrapz(rows_z_u, axis=0, initial=0)[::-1]
    rows_z_d = grad_zh[h_c:]
    rows_h_d = coord_h[h_c:]
    # rows_d_int = integrate.cumtrapz(rows_z_d, x=rows_h_d, axis=0, initial=0)
    rows_d_int = integrate.cumtrapz(rows_z_d, axis=0, initial=0)
    rows_int = np.concatenate((rows_u_int, rows_d_int[1:]), axis=0)
    xy_int = rows_int + row0_int[None]

    # fusion
    mask_int = ~np.isnan(yx_int) & ~np.isnan(xy_int) & mask
    mask_yx = ~np.isnan(yx_int) & np.isnan(xy_int) & mask
    mask_xy = np.isnan(yx_int) & ~np.isnan(xy_int) & mask

    yx_int[np.isnan(yx_int)] = 0
    xy_int[np.isnan(xy_int)] = 0
    # coord_h = mask_int * coord_h
    # coord_w = mask_int * coord_w
    # yx_int += 20
    # xy_int += 20
    coord_z = mask_int * yx_int + mask_yx * yx_int + mask_xy * xy_int

    return coord_h, coord_w, coord_z, mask_int * 1
