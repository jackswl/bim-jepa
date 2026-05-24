# Self-supervised learning for BIM element classification using a joint embedding predictive architecture

Created by [Jack Wei Lun Shi](https://jackswl.github.io/)\*, [Wawan Solihin](https://cde.nus.edu.sg/cee/staff/wawan-solihin/), Yufeng Weng, [Yimin Zhao](https://ztony0712.github.io/), [Leong Hien Poh](https://scholar.google.com/citations?user=xZ3x56EAAAAJ&hl=en), [Justin K.W. Yeoh](https://scholar.google.com/citations?user=m9LF49sAAAAJ&hl=en)

[[Automation in Construction]](https://jackswl.github.io/bim-jepa/) [[Project Page]](https://jackswl.github.io/bim-jepa/) [[Model Weights]](#pretrained-models)

This repository contains BIM-JEPA implementation for __Self-supervised learning for BIM element classification using a joint embedding predictive architecture__.

The development of scalable models for automated Building Information Modeling (BIM) element classification is hindered by the reliance on supervised learning, which requires expensive and laborious manual data annotation. This paper introduces a pre-trained model that leverages a Joint Embedding Predictive Architecture for self-supervised learning on unlabeled 3D point cloud representations of individual BIM elements. By predicting the latent representations of masked regions of element geometry, the proposed model learns rich geometric features that achieve competitive accuracy on a downstream classification task, outperforming existing supervised methods without heavy data augmentation, while excelling in data-scarce scenarios. This paper mitigates the data annotation bottleneck and establishes a path toward developing a foundation model for BIM geometry, enabling more scalable, data-efficient, and generalizable representation learning in the Architecture, Engineering, and Construction domain.

## <a id="pretrained-models"></a>Pre-trained / Fine-tuned Models

All model weights are available on HuggingFace.

|model| dataset | config | url|
| :---: | :---: | :---: |  :---: | 
|BIM-JEPA-pretrained| IFC-884; IFCNet; BIMGEOM |  [config](BIM-JEPA/configs/BIM-JEPA/pretraining/combined_pretrain_original.yaml)|  [HuggingFace](https://huggingface.co/llama2thedog/BIM-JEPA-pretrained)|

|model| dataset  | Overall Acc | Mean Acc | config | url|
| :---:| :---: | :---: |  :---: | :---: | :---: |
|BIM-JEPA-IFCNetCore| IFCNetCore | 89.37 | 86.63 |  [config](BIM-JEPA/configs/BIM-JEPA/classification/ifcnet_classification.yaml) | [HuggingFace](https://huggingface.co/llama2thedog/BIM-JEPA-finetuned-ifcnetcore) |
|BIM-JEPA-BIMGEOM| BIMGEOM | 92.43 | 89.53 |[config](BIM-JEPA/configs/BIM-JEPA/classification/bimgeom_classification.yaml) | [HuggingFace](https://huggingface.co/llama2thedog/BIM-JEPA-finetuned-bimgeom) |


## Usage

### Requirements
- PyTorch >= 2.4.1
- python == 3.11
- CUDA >= 12.1
- torchvision
- PyTorch3D

### Conda Installation
Option A (Recommended) -> You can create conda environment using:
```
conda env create -f environment.yml
conda activate bimjepa
```

Option B -> Create environment:
```
conda create -n bimjepa \
    python=3.11 \
    pytorch=2.4.1 \
    torchvision=0.19.1 \
    pytorch-cuda=12.1 \
    cudatoolkit \
    -c pytorch -c nvidia -y
```
After that, install PyTorch3D (https://github.com/facebookresearch/pytorch3d):
```
export FORCE_CUDA=1
conda install pytorch3d::pytorch3d
```
Finally, install the remaining miscellaneous/visualization packages:
```
pip install transformers accelerate "pytorch-lightning>=2.0" "jsonargparse[signatures]" trimesh scikit-learn h5py matplotlib wandb timm lightning-bolts fvcore pandas seaborn plotly
```

### Dataset
The details of raw data can be found in [DATASET.md](./DATASET.md). After downloading the raw data, please run [data_convert.ipynb](./data_convert.ipynb) to convert all raw data into NPY point clouds.

### Pre-trained Weights
All checkpoints are hosted on HuggingFace (see the [table above](#pretrained-models)). First install the HuggingFace CLI:
```
pip install -U "huggingface_hub[cli]"
```

To fine-tune from the pre-trained encoder, download the pre-trained checkpoint into `BIM-JEPA/pretrained/` (this is the default path expected by the classification configs):
```
cd BIM-JEPA
huggingface-cli download llama2thedog/BIM-JEPA-pretrained \
    bim_jepa_pretrained.ckpt --local-dir pretrained
```

To run inference / evaluation with a fine-tuned classifier, download the corresponding fine-tuned checkpoint, e.g.:
```
huggingface-cli download llama2thedog/BIM-JEPA-finetuned-ifcnetcore \
    bim_jepa_finetuned_ifcnetcore.ckpt --local-dir pretrained
```

### Training and Evaluation
All commands below are run from the `BIM-JEPA/` directory.

Fine-tune on IFCNetCore:
```
python -m bimjepa.tasks.classification fit \
    -c configs/BIM-JEPA/classification/ifcnet_classification.yaml
```

Fine-tune on BIMGEOM:
```
python -m bimjepa.tasks.classification fit \
    -c configs/BIM-JEPA/classification/bimgeom_classification.yaml
```

Pre-train from scratch (multi-GPU, expects all five pre-training datasets prepared as in [DATASET.md](./DATASET.md)):
```
python -m bimjepa fit \
    -c configs/BIM-JEPA/pretraining/combined_pretrain_original.yaml
```

For HPC users, example PBS scripts are provided in [`BIM-JEPA/compute/`](./BIM-JEPA/compute/) — see [`BIM-JEPA/compute/README.md`](./BIM-JEPA/compute/README.md) for details.

### Quick Demo
For a no-install Colab walkthrough that downloads a fine-tuned model and runs inference on sample point clouds, see [`BIM_JEPA_demo.ipynb`](./BIM_JEPA_demo.ipynb).


## License
MIT License

## Citation
If you find our work useful in your research, please consider citing: 
```
in progress
```

## Acknowledgements
We sincerely thank the authors of Point-JEPA, SpaRSE-BIM/IFCNet, and BIMGEOM for making their code/data and models publicly available, which served as the foundation for this work. If you use our work, please also consider citing these papers.