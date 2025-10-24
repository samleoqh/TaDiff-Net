# SAILOR Dataset Preprocessing

This repository contains preprocessing scripts for the SAILOR (brain tumor MRI) longitudinal dataset. The script processes multi-modal MRI scans and segmentation masks, preparing them for TaDiff-Net models. [SAILOR raw data download link](https://search.kg.ebrains.eu/instances/cae85bcb-8526-442d-b0d8-a866425efff8)

## Dataset Structure

The SAILOR dataset follows this structure:ß

```
sailor-raw/
└── sailor_info.csv # Patient metadata (patient_id, age, num_sessions, intervals)
└── sub-{01}/
    ├── ses-{01}/
    │   ├── EdemaMask-CL.nii.gz              # Edema segmentation
    │   ├── ContrastEnhancedMask-CL.nii.gz   # Enhancing tumor segmentation
    │   ├── T1.nii.gz                         # T1-weighted MRI
    │   ├── T1c.nii.gz                        # T1 contrast-enhanced MRI
    │   ├── Flair.nii.gz                      # FLAIR MRI
    │   └── T2.nii.gz                         # T2-weighted MRI
    ├── ses-{02}/
    │   └── ...
    ├── age-years.txt
    ├── DoseMap.nii.gz
    ├── intervals-days.txt
    └── overall-survival-months.txt
```

## Features

### Data Processing
- **NIfTI Loading**: Reads `.nii.gz` files with reorientation support
- **Intensity Normalization**: Non-zero voxel normalization with outlier clipping
- **Mask Merging**: Combines edema and enhancing tumor masks
- **Temporal Information**: Extracts time intervals and treatment schedules
- **Batch Processing**: Processes all patients and sessions automatically

### Output Format
For each patient, the script generates:
- `{patient_id}_image.npy`: Shape `(M×T, H, W, D)` - All modalities and timepoints
- `{patient_id}_label.npy`: Shape `(T, H, W, D)` - Segmentation masks
- `{patient_id}_days.npy`: Shape `(T,)` - Cumulative days from baseline
- `{patient_id}_treatment.npy`: Shape `(T,)` - Treatment type per timepoint

Where:
- `M = 4` modalities (T1, T1c, FLAIR, T2)
- `T` = number of timepoints/sessions (varies per patient)
- `H, W, D` = spatial dimensions (typically 240×240×155)

## Installation

### Requirements
```bash
pip install numpy pandas nibabel torch reorient_nii
```


## Usage

### Basic Usage

1. **Set the data path**:
   ```python
   ROOT = '/path/to/sailor-raw'
   ```

2. **Run preprocessing**:
   ```bash
   python preproc_prepare_data.py
   ```

3. **Output location**:
   By default, processed data is saved to `./sailor`

### Advanced Usage

```python
from preproc_prepare_data import save_session_data, get_file_dict

# Process specific patients
custom_patients = ['sub-01', 'sub-02', 'sub-03']
file_dict = get_file_dict(patient_ids=custom_patients)
save_session_data(file_dict, save_path='./custom_output')

# Load preprocessed data
import numpy as np
images = np.load('data/sailor/sub-01_image.npy')
labels = np.load('data/sailor/sub-01_label.npy')
days = np.load('data/sailor/sub-01_days.npy')
treatment = np.load('data/sailor/sub-01_treatment.npy')
```

## Data Specifications

### MRI Modalities
| Index | Modality | Description |
|-------|----------|-------------|
| 0 | T1 | T1-weighted MRI |
| 1 | T1c | T1 contrast-enhanced |
| 2 | FLAIR | Fluid-attenuated inversion recovery |
| 3 | T2 | T2-weighted MRI |

### Segmentation Labels
| Value | Label | Description |
|-------|-------|-------------|
| 0 | Background | Non-tumor tissue |
| 1 | Edema | Peritumoral edema |
| 2 | Necrotic | Necrotic core (reserved) |
| 3 | Enhancing | Enhancing tumor |

### Treatment Types
| Value | Treatment | Description |
|-------|-----------|-------------|
| 0 | CRT | Chemoradiotherapy (sessions 0-3) |
| 1 | TMZ | Temozolomide (sessions 4+) |

## Patient Cohort

### Statistics
- **Total patients**: 27 (sub-01 to sub-27)
- **Valid patients**: 25 (after QC) sicne some sessions with co-registration issues or missing modalities or missing all CL masks


## Preprocessing Pipeline

### 1. Image Normalization
```python
def nonzero_norm_image(image, clip_percent=0.1):
    # 1. Clip outliers at percentiles
    # 2. Z-score normalization
    # 3. Scale to [0, 1]
```

**Parameters**:
- `clip_percent=0.2`: Clips bottom/top 0.2% intensities
- Only normalizes non-zero voxels (preserves background)

### 2. Orientation
- Target orientation: **PLI** (Posterior-Left-Inferior)
- Ensures consistent spatial alignment

### 3. Mask Merging
- Combines `EdemaMask-CL` and `ContrastEnhancedMask-CL`
- Priority: Enhancing tumor (label 3) overwrites edema (label 1)


## File Structure

```
.
├── preproc_prepare_data.py    # Main preprocessing script
├── README.md                  # This file
└── data/
    └── sailor/            # Output directory
        ├── sub-01_image.npy
        ├── sub-01_label.npy
        ├── sub-01_days.npy
        └── sub-01_treatment.npy
```

## Metadata (sailor_info.csv)

Required CSV format:
```csv
patients	age	survival_months	num_ses	interval_days
sub-01	64.25479452	33.50819672	6	[13, 15, 14, 34, 28]
...
```

## Common Issues

### Issue: "Image has no non-zero values"
**Cause**: Empty or corrupted NIfTI file  
**Solution**: Check file integrity, exclude patient if persistent

### Issue: "std of image is zero"
**Cause**: Constant-valued image (rare)  
**Solution**: Image is likely corrupted, skip this modality/session

### Issue: Missing files
**Cause**: Incomplete dataset  
**Solution**: Ensure all required files exist in `KEY_FILENAMES`

## Output Validation

After preprocessing, verify your data:

```python
import numpy as np

# Load data
img = np.load('data/sailor_npy/sub-01_image.npy')
lbl = np.load('data/sailor_npy/sub-01_label.npy')

# Check shapes
print(f"Image: {img.shape}")  # e.g., (24, 240, 240, 155) = 4 modalities × 6 sessions
print(f"Label: {lbl.shape}")  # e.g., (6, 240, 240, 155) = 6 sessions

# Check value ranges
print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")  # Should be [0, 1]
print(f"Label values: {np.unique(lbl)}")  # Should be [0, 1, 3]

# Check for NaN/Inf
assert not np.isnan(img).any(), "NaN detected in images"
assert not np.isinf(img).any(), "Inf detected in images"
```

