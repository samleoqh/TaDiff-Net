# TaDiff-Net
This repository contains code for the paper "[Treatment-aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction](https://doi.org/10.1109/TMI.2025.3533038)".

<div style="display: flex; align-items: unsafe center;">
    <img src="demo_1.gif" alt="Demo GIF" style="height: 260px;">
    <img src="tadiff_concept.png" alt="Concept Image" style="width: 650px;">
</div>

## Overview
We propose a novel end-to-end network capable of future predictions of tumor masks and multi-parametric magnetic resonance images (MRI) of how the tumor will look at any future time points for different treatment plans. Our approach is based on cutting-edge diffusion probabilistic models and deep-segmentation neural networks. We included sequential multi-parametric MRI and treatment information as conditioning inputs to guide the generative diffusion process as well as a joint segmentation process. This allows for tumor growth estimates and realistic MRI generation at any given treatment and time point. We trained the model using real-world [postoperative longitudinal MRI data](https://search.kg.ebrains.eu/instances/cae85bcb-8526-442d-b0d8-a866425efff8) with glioma tumor growth trajectories represented as tumor segmentation maps over time. The model demonstrates promising performance across various tasks, including generating high-quality multi-parametric MRI with tumor masks, performing time-series tumor segmentations, and providing uncertainty estimates. Combined with the treatment-aware generated MRI, the tumor growth predictions with uncertainty estimates can provide useful information for clinical decision-making.

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
@article{liu2025treatment,
  title={Treatment-aware diffusion probabilistic model for longitudinal MRI generation and diffuse glioma growth prediction},
  author={Liu, Qinghui and Fuster-Garcia, Elies and Hovden, Ivar Thokle and MacIntosh, Bradley J and Gr{\o}dem, Edvard OS and Brandal, Petter and Lopez-Mateu, Carles and Sederevi{\v{c}}ius, Donatas and Skogen, Karoline and Schellhorn, Till and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```

