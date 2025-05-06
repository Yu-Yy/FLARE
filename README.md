# FLARE: Fixed-Length Dense Fingerprint Representation

This repository accompanies our paper:

**Fixed-Length Dense Fingerprint Representation**  
Zhiyu Pan, Xiongjin Guan, Yongjie Duan, Jianjiang Feng, and Jie Zhou  
*Under review at IEEE Transactions on Information Forensics and Security (TIFS)*

---

## 🔍 Overview

**FLARE** is a fingerprint recognition framework designed for robust and efficient matching using fixed-length dense descriptors. It integrates:

- 🧠 **Dense Descriptor** (to be released upon acceptance)
- 🧩 **Pose-Aware Alignment** (via [FLARE-Align](https://github.com/XiongjunGuan/DRACO))
- 🧪 **Robust Enhancement** (via [FLARE-Enh](https://github.com/Yu-Yy/FLARE_ENH))

This repository currently contains references and links to the **pose estimation** and **enhancement** modules only.  
The core code for descriptor extraction and matching will be released **after the paper is accepted**.

---

## 📦 Available Components

| Module           | Description                                | Link                                         |
|------------------|--------------------------------------------|----------------------------------------------|
| FLARE-Enh        | Fingerprint enhancement (UNetEnh, PriorEnh) | [https://github.com/Yu-Yy/FLARE_ENH](https://github.com/Yu-Yy/FLARE_ENH) |
| FLARE-Align      | Pose estimation and alignment (DRACO)       | [https://github.com/XiongjunGuan/DRACO](https://github.com/XiongjunGuan/DRACO) |
| FLARE-Desc       | Fixed-length Dense Descriptor extraction and matching          | **Coming soon upon paper acceptance**        |

---

## 📄 Citation

Please cite the following if you reference FLARE or use our enhancement modules:

```

@article{Pan2025FLARE,
title     = {Fixed-Length Dense Fingerprint Representation},
author    = {Zhiyu Pan and Xiongjin Guan and Yongjie Duan and Jianjiang Feng and Jie Zhou},
journal   = {IEEE Transactions on Information Forensics and Security (under review)},
year      = {2025}
}

```

---

## ⚠️ License & Usage Notice

The released code and models are provided **for academic research and educational use only**.  
**Commercial use is not permitted.**

---
## 📬 Contact
For questions or collaboration, please contact:  
📧 Zhiyu Pan (pzy20@mails.tsinghua.edu.cn)