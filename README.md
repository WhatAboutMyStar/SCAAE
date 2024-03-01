# SCAAE
- Discovering Dynamic Functional Brain Networks via Spatial and Channel-wise Attention [arxiv:2205.09576](https://arxiv.org/abs/2205.09576) 
- Mapping dynamic spatial patterns of brain function with spatial-wise attention [paper link](https://iopscience.iop.org/article/10.1088/1741-2552/ad2cea)
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
The result in ADHD-200 dataset shown in [constructFBN.ipynb](./constructFBN.ipynb)

The result in task-based fMRI dataset shown in [task-gambling-avg-group-wise-40.ipynb](./task-gambling-avg-group-wise-40.ipynb). 

We use the [Baseline](https://github.com/SNNUBIAI/Baseline) developed by our [SNNUBIAI Lab](https://github.com/SNNUBIAI/) for evaluation. 

## Citing SCAAE
```
@article{liu2022discovering,
  title={Discovering Dynamic Functional Brain Networks via Spatial and Channel-wise Attention},
  author={Liu, Yiheng and Ge, Enjie and He, Mengshen and Liu, Zhengliang and Zhao, Shijie and Hu, Xintao and Zhu, Dajiang and Liu, Tianming and Ge, Bao},
  journal={arXiv preprint arXiv:2205.09576},
  year={2022}
}

@article{liu2024mapping,
  title={Mapping dynamic spatial patterns of brain function with spatial-wise attention},
  author={Liu, Yiheng and Ge, Enjie and He, Mengshen and Liu, Zhengliang and Zhao, Shijie and Hu, Xintao and Qiang, Ning and Zhu, Dajiang and Liu, Tianming and Ge, Bao},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

## Related Work
- [STCA](https://github.com/SNNUBIAI/STCAE)
```
@inproceedings{liu2023spatial,
  title={Spatial-Temporal Convolutional Attention for Mapping Functional Brain Networks},
  author={Liu, Yiheng and Ge, Enjie and Qiang, Ning and Liu, Tianming and Ge, Bao},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}

@article{liu2024spatial,
  title={Spatial-temporal convolutional attention for discovering and characterizing functional brain networks in task fMRI},
  author={Liu, Yiheng and Ge, Enjie and Kang, Zili and Qiang, Ning and Liu, Tianming and Ge, Bao},
  journal={NeuroImage},
  pages={120519},
  year={2024},
  publisher={Elsevier}
}
```
