# TaDiff-Net
This repository contains code for the paper "[Treatment-aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction](https://doi.org/10.1109/TMI.2025.3533038)".

<div style="display: flex; align-items: unsafe center;">
    <img src="demo_1.gif" alt="Demo GIF" style="height: 260px;">
    <img src="tadiff_concept.png" alt="Concept Image" style="width: 650px;">
</div>

## Overview
We propose a novel end-to-end network capable of future predictions of tumor masks and multi-parametric magnetic resonance images (MRI) of how the tumor will look at any future time points for different treatment plans. Our approach is based on cutting-edge diffusion probabilistic models and deep-segmentation neural networks. We included sequential multi-parametric MRI and treatment information as conditioning inputs to guide the generative diffusion process as well as a joint segmentation process. This allows for tumor growth estimates and realistic MRI generation at any given treatment and time point. We trained the model using real-world [postoperative longitudinal MRI data](https://search.kg.ebrains.eu/instances/cae85bcb-8526-442d-b0d8-a866425efff8) with glioma tumor growth trajectories represented as tumor segmentation maps over time. The model demonstrates promising performance across various tasks, including generating high-quality multi-parametric MRI with tumor masks, performing time-series tumor segmentations, and providing uncertainty estimates. Combined with the treatment-aware generated MRI, the tumor growth predictions with uncertainty estimates can provide useful information for clinical decision-making.


## Project Structure

```
TaDiff-Net/
├── config/
│   └── test_config.py      # Configuration settings
├── src/
│   ├── data/
│   │   └── data_loader.py  # Data loading and preprocessing
│   ├── evaluation/
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── ssim.py         # SSIM implementation
│   ├── net/
│   │   └── diffusion.py    # Diffusion model implementation
│   ├── visualization/
│   │   └── visualizer.py   # Visualization utilities
│   └── tadiff_model.py     # Main model implementation
├── ckpt/                   # Model checkpoints
├── data/                   # Data directory
├── test.py                 # Testing script
├── inference.py               # Inference script
└── README.md              # This file
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TaDiff-Net.git
cd TaDiff-Net
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your data files in the `data/` directory
   - Data should be organized as follows:
     ```
     data/
     ├── {patient_id}_image.npy    # Image data
     ├── {patient_id}_label.npy    # Ground truth labels (for testing)
     ├── {patient_id}_days.npy     # Time points
     └── {patient_id}_treatment.npy # Treatment information
     ```

## Configuration

The configuration settings are defined in `config/test_config.py`. Key parameters include:

- Model parameters (channels, heads, etc.)
- Test/inference parameters (diffusion steps, number of samples)
- Data paths and patient IDs
- Visualization settings

## Usage

### Testing

To evaluate the model with ground truth data:

```bash
python test.py
```

This will:
1. Load the model and test data
2. Generate predictions for each patient and slice
3. Calculate evaluation metrics
4. Save results and visualizations

Output will be saved in the configured save path with:
- Evaluation metrics in CSV format
- Prediction visualizations
- Uncertainty maps
- Ground truth comparisons

### Inference

To generate predictions without ground truth:

```bash
python inference.py --input_day 20 --input_treatment 1
```

This will:
1. Load the model and input data
2. Generate predictions for input patient and target slice
3. Save ensemble predictions and plot uncertainty maps

Output structure:
```
save_path/
├── p-{patient_id}/ses-{target_sess}/day-{target_day}
│   ├── treatment-{0/1}/
│   │   ├── prediction-slice-{}.npy
│   │   ├── segmentation-slice-{}.npy
│   │   ├── ses-{}-slice-{}-pred_{t1,t1c,flair}.png    # Average predictions (generated 3 modal MRI slices)
│   │   ├── ses-{}-slice-{}-pred_mask.png            # predicted target tumor segmentation mask
│   │   ├── ses-{}-slice-{}-uncertainty_{t1,t1c,flair}.png # predictied 3 modal image uncertainty
│   │   ├── ses-{}-slice-{}-uncertainty_mask.png     # target tumor mask uncertainty
│   
```

## Model Details

TaDiff-Net uses a diffusion-based approach for tumor growth prediction and segmentation. Key features:

- Multi-modal input (T1, T1c, FLAIR)
- Temporal modeling of tumor growth
- Treatment-aware predictions
- Uncertainty estimation through multiple samples

## Evaluation Metrics

The model is evaluated using:
- Dice coefficient (at multiple thresholds)
- Mean Absolute Error (MAE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Relative Absolute Volume Difference (RAVD)

## Visualization

The visualization module provides:
- Prediction overlays
- Uncertainty maps
- Contour visualization
- Multi-modal image display


## Citation

If you find this code helps in your work, please cite:
```
@ARTICLE{10851394,
  author={Liu, Qinghui and Fuster-Garcia, Elies and Thokle Hovden, Ivar and MacIntosh, Bradley J. and Grødem, Edvard O. S. and Brandal, Petter and Lopez-Mateu, Carles and Sederevičius, Donatas and Skogen, Karoline and Schellhorn, Till and Bjørnerud, Atle and Eeg Emblem, Kyrre},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Treatment-Aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction}, 
  year={2025},
  volume={44},
  number={6},
  pages={2449-2462},
  keywords={Tumors;Magnetic resonance imaging;Predictive models;Brain modeling;Uncertainty;Data models;Probabilistic logic;Diffusion processes;Diffusion models;Computational modeling;Diffuse glioma;longitudinal MRI;diffusion probabilistic model;tumor growth prediction;deep learning},
  doi={10.1109/TMI.2025.3533038}}
```
