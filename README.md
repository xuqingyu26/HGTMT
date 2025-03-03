# HGTMT
PyTorch implementation of  "Heterogeneous Graph Transformer for Multiple Tiny Object Tracking in RGB-T Videos", IEEE Transactions on MultiMedia，[paper](https://arxiv.org/abs/2412.10861). Feel free to contact me (xuqingyu@nudt.edu.cn) if any questions. Please star the repository~~
## Dataset:
Download the dataset at [BaiduYun](https://pan.baidu.com/s/1noAoDpJGc3AFF_4gnBH2wg?pwd=3crl)[3crl].
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
