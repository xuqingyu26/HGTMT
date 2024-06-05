# HGTMT
PyTorch implementation of  "Heterogeneous Graph Transformer for Multiple Tiny Object Tracking in RGB-T Videos", IEEE Transactions on MultiMedia. Feel free to contact me (xuqingyu@nudt.edu.cn) if any questions. Please star the repository~~

## Dataset: VT-Tiny-MOT 
* Download the dataset from [Baidu Drive](https://pan.baidu.com/s/1VFUkv7h1US5Xgb_XpAInXg?pwd=VTMT) (Key: VTMT) and unzip them to `./data` 
### Note:
* The multiple tiny object tracking dataset is composed of two modalities: visible and thermal, and is well aligned.
## Code: 
 * Create the working environment through environment.yml
 * Download the transformer pvtv2 backbone from [PVTv2](https://github.com/whai362/PVT).
 * Run training/main_RGBT-Tiny_graph_gnnloss.py for training.
 * Run tracking/RGBT-Tiny_private_graph_Track2_crossmodal.py for tracking.
## Acknowledgement:
 * This repository benefits a lot from [TransCenter](https://github.com/yihongXU/TransCenter) and [GSDT](https://github.com/yongxinw/GSDT).
## Cite:
```
@article{xu2024heterogeneous,
  title={Heterogeneous Graph Transformer for Multiple Tiny Object Tracking in RGB-T Videos},
  author={Xu, Qingyu and Wang, Longguang and Sheng, Weidong and Wang, Yingqian and Xiao, Chao and Ma, Chao and An, Wei},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```
