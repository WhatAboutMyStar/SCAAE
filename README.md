# SCAAE
 Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022)
 - Discovering Dynamic Functional Brain Networks via Spatial and Channel-wise Attention [arxiv:2205.09576](https://arxiv.org/abs/2205.09576)
## Download Training Data
A simple way to download part of ADHD-200 datasets is to use nilearn. These data can be used as training data.
```
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd(n_subjects=40, "./data/")
```
You can make the dataset by:
```
from dataset import LoadADHD200
data = LoadADHD200(img_path="./data/adhd/data/", 
                    mask_path="./data/ADHD200_mask_152_4mm.nii.gz",
                    save_fmri=True, 
                    save_path="./data/adhd200.npy")
```
More preprocessed ADHD-200 data can be accessed here [ADHD-200 Preprocessed](http://preprocessed-connectomes-project.org/adhd200/).

## Training
- Training model
```
chmod +x train.sh
./train.sh
```

- Tensorboard
```
tensorboard --logdir=./logdir/
```

## Result
The result shown in [constructFBN.ipynb](./constructFBN.ipynb)

## Citing SCAAE
```
@article{liu2022discovering,
  title={Discovering Dynamic Functional Brain Networks via Spatial and Channel-wise Attention},
  author={Liu, Yiheng and Ge, Enjie and He, Mengshen and Liu, Zhengliang and Zhao, Shijie and Hu, Xintao and Zhu, Dajiang and Liu, Tianming and Ge, Bao},
  journal={arXiv preprint arXiv:2205.09576},
  year={2022}
}
```
