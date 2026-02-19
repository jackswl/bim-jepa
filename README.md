# BIM-JEPA: Self-supervised learning for BIM element classification using a joint embedding predictive architecture

Created by [Jack Wei Lun Shi](https://jackswl.github.io/)\*, [Wawan Solihin](https://cde.nus.edu.sg/cee/staff/wawan-solihin/), Yufeng Weng, [Yimin Zhao](https://ztony0712.github.io/), [Leong Hien Poh](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1), [Justin K.W. Yeoh](https://scholar.google.com/citations?user=m9LF49sAAAAJ&hl=en)

[[Automation in Construction]](https://jackswl.github.io/bim-jepa/) [[Project Page]](https://jackswl.github.io/bim-jepa/) [[Model Weights In Progress]](#pretrained-models)

This repository contains BIM-JEPA implementation for __BIM-JEPA: Self-supervised learning for BIM element classification using a joint embedding predictive architecture__ (Under Review).

All training code and weights will be released upon acceptance of the paper.

## Pretrained Models
More details will be released upon acceptance of paper.

## Usage

### Requirements
- PyTorch >= 2.4.1
- python == 3.11
- CUDA >= 12.1
- torchvision
- pytorch3D

### Conda Installation
#### Option A
Create environment:
```
conda create -n bimjepa \
    python=3.11 \
    pytorch=2.4.1 \
    torchvision=0.19.1 \
    pytorch-cuda=12.1 \
    cudatoolkit \
    -c pytorch -c nvidia -y
```
After that, install pytorch3D (https://github.com/facebookresearch/pytorch3d):
```
export FORCE_CUDA=1
conda install pytorch3d::pytorch3d
```
Finally, install the remaining miscellaneous/visualization packages:
```
pip install transformers accelerate "pytorch-lightning>=2.0" "jsonargparse[signatures]" trimesh scikit-learn h5py matplotlib wandb timm lightning-bolts fvcore pandas seaborn plotly
```
#### Option B (recommendated)
You can also simply create conda environment using:
```
conda env create -f environment.yml
conda activate bimjepa
```

### Dataset
The details of raw datasets can be found in [DATASET.md](./DATASET.md). More details for datasets will be released upon acceptance of paper.


## License
MIT License

## Citation
If you find our work useful in your research, please consider citing: 
```
in progress
```
