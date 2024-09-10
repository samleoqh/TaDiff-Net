# TaDiff-Net
This repository contains code for the paper "[Treatment-aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction](https://arxiv.org/abs/2309.05406)", which is currently under review. Once the paper is published, we will release the code and data in accordance with company policies.

<div style="display: flex; align-items: unsafe center;">
    <img src="demo_1.gif" alt="Demo GIF" style="height: 260px;">
    <img src="tadiff_concept.png" alt="Concept Image" style="width: 650px;">
</div>

## Overview
In this paper, we present a novel end-to-end network capable of generating future tumor masks and realistic MRIs of how the tumor will look at any future time points for different treatment plans. Our approach is based on cutting-edge diffusion probabilistic models and deep-segmentation neural networks. We included sequential multi-parametric magnetic resonance images (MRI) and treatment information as conditioning inputs to guide the generative diffusion process. This allows for tumor growth estimates at any given time point. We trained the model using real-world [postoperative longitudinal MRI data](https://search.kg.ebrains.eu/instances/cae85bcb-8526-442d-b0d8-a866425efff8) with glioma tumor growth trajectories represented as tumor segmentation maps over time. The model has demonstrated promising performance across a range of tasks, including the generation of high-quality synthetic MRIs with tumor masks, time-series tumor segmentations, and uncertainty estimates. Combined with the treatment-aware generated MRIs, the tumor growth predictions with uncertainty estimates can provide useful information for clinical decision-making.

## Code Structure

The code is organized as follows:

- `data_preprocessing/`: Scripts for preprocessing MRI data.
- `model_training/`: Implementation of the proposed diffusion probabilistic model and training procedures.
- `evaluation/`: Evaluation scripts for assessing model performance.
- `visualization/`: Tools for visualizing MRI data and model predictions.
- `utils/`: Utility functions used throughout the codebase.


## Citation
Please consider citing [our work](https://arxiv.org/abs/2309.05406) if you find it helps you in your work. 
```
@article{liu2023treatment,
  title={Treatment-aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction},
  author={Liu, Qinghui and Fuster-Garcia, Elies and Hovden, Ivar Thokle and Sederevicius, Donatas and Skogen, Karoline and MacIntosh, Bradley J and Gr{\o}dem, Edvard and Schellhorn, Till and Brandal, Petter and Bj{\o}rnerud, Atle and others},
  journal={arXiv preprint arXiv:2309.05406},
  year={2023}
}
```

