"""
TaDiff Model Inference Script

This script performs inference using the TaDiff model without requiring ground truth.
It generates predictions for medical images and saves the results.

Key Features:
- Loads pre-trained TaDiff model
- Processes input medical images (MRI scans)
- Supports multiple diffusion sampling methods
- Generates predictions for specified target sessions
- Saves predictions as numpy arrays and visualizations

Usage:
    python inference.py --patient_ids 17 --diffusion_steps 50 --num_samples 4
"""

import os
import argparse
import torch
import numpy as np
from typing import List, Dict, Optional

from src.tadiff_model import Tadiff_model
from src.net.diffusion import GaussianDiffusion
from src.data.data_loader import load_data
from src.visualization.visualizer import save_visualization_results, create_directory
from src.utils.image_processing import prepare_image_batch, create_noise_tensor

def get_input_files(data_root: str = "./data/lumiere",
                   patient_ids: Optional[List[str]] = None,
                   prefix: str = '') -> List[Dict[str, str]]:
    """
    Get list of input files for specified patients.
    
    Args:
        data_root: Root directory containing .npy files
        patient_ids: List of patient IDs to process
        prefix: Optional filename prefix
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with keys:
            - 'image': Path to image .npy file
            - 'days': Path to days .npy file
            - 'treatment': Path to treatment .npy file
            
    Raises:
        FileNotFoundError: If required .npy files are missing
    """
    npz_keys = ['image', 'label','days', 'treatment']
    input_files = []
    
    for patient_id in patient_ids:
        file_dict = {}
        for key in npz_keys:
            file_path = os.path.join(data_root, f'{prefix}{patient_id}_{key}.npy')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
            file_dict[key] = file_path
        input_files.append(file_dict)
        
    return input_files

def process_session(
    patient_id: str,
    batch: Dict[str, torch.Tensor],
    model: Tadiff_model,
    device: torch.device,
    save_path: str,
    input_days: torch.Tensor,
    input_treatments: torch.Tensor,
    slice_idx: int,
    diffusion_steps: int = 600,
    num_samples: int = 5
) -> None:
    """
    Process a single patient session through inference pipeline.
    
    Args:
        patient_id: Unique patient identifier
        session_idx: Index of the session being processed
        batch: Dictionary containing:
            - 'image': Input scans [1, num_sessions, C, H, W, D]
            - 'days': Treatment day values [1, num_sessions]
            - 'treatment': Treatment codes [1, num_sessions]
        model: Loaded Tadiff_model instance
        device: Target device
        save_path: Base directory for saving results
        diffusion_steps: Number of diffusion steps
        num_samples: Number of predictions to generate
        target_idx: Index of target session for prediction
    """
    # Extract and prepare tensors
    images = batch['image'].to(device)
    hist_days = batch['days'].to(device)
    hist_treatments = batch['treatment'].to(device)
    
    # Concatenate historical and input days/treatments
    days = torch.cat([hist_days, input_days.unsqueeze(0)], dim=1)
    treatments = torch.cat([hist_treatments, input_treatments.unsqueeze(0)], dim=1)
    
    # Prepare image batch (add empty channel for future session)
    n_sessions = images.shape[1]//4 #+ 1  # +1 for future session
    print('n_sessions', n_sessions)
    images = prepare_image_batch(images, n_sessions)
    
    # Create session directory
    target_session_idx = images.shape[1]//4 #- 1  # Last session is target
    session_path = os.path.join(save_path, f'p-{patient_id}', f'-ses-{target_session_idx:02d}')
    create_directory(session_path)
    process_slice(
        slice_idx=slice_idx,
        images=images,
        days=days,
        treatments=treatments,
        model=model,
        device=device,
        session_path=session_path,
        diffusion_steps=diffusion_steps,
        num_samples=num_samples,
        target_idx=target_session_idx
    )

def process_slice(
    slice_idx: int,
    images: torch.Tensor,
    days: torch.Tensor,
    treatments: torch.Tensor,
    model: Tadiff_model,
    device: torch.device,
    session_path: str,
    diffusion_steps: int,
    num_samples: int,
    target_idx: int
) -> None:
    """
    Process a single 2D slice through inference pipeline.
    
    Args:
        slice_idx: Z-index of the slice being processed
        images: Input scans [1, num_sessions, C, H, W, D]
        days: Treatment day values [1, num_sessions]
        treatments: Treatment codes [1, num_sessions]
        model: Loaded Tadiff_model instance
        device: Target device
        session_path: Directory for saving results
        diffusion_steps: Number of diffusion steps
        num_samples: Number of predictions to generate
        target_idx: Index of target session for prediction
    """
    # Prepare data - always use last 4 sessions (including new target)
    slice_indices = [slice_idx] * num_samples
    n_sessions = target_idx #days.shape[1]
    session_indices = np.array([
        n_sessions - 3,
        n_sessions - 2,
        n_sessions - 1,
        n_sessions -1
    ])
    session_indices[session_indices < 0] = 0
    session_indices = list(session_indices)

    # Extract relevant slices
    seq_imgs = images[0, session_indices, :, :, :, :]
    seq_imgs = seq_imgs[:, :, :, :, slice_indices].permute(4, 0, 1, 2, 3)
    
    # Create noise and prepare target images
    noise = create_noise_tensor(num_samples, 3, images.shape[3], images.shape[4], device)
    x_t = seq_imgs.clone()
    x_0 = []
    
    # Set up target indices
    i_tg = target_idx * torch.ones((len(slice_indices),), dtype=torch.int8).to(device)
    
    # Prepare input tensors
    for i, j in zip(range(num_samples), i_tg):
        x_0.append(seq_imgs[[i], j, :, :, :])
        x_t[i, j, :, :, :] = noise[i, :, :, :]
    x_0 = torch.cat(x_0, dim=0)
    x_t = x_t.reshape(num_samples, len(session_indices) * 3, images.shape[3], images.shape[4])
    
    # Prepare condition tensors
    daysq = days[0, session_indices].repeat(num_samples, 1)
    treatments_q = treatments[0, session_indices].repeat(num_samples, 1)
    
    # Run diffusion
    diffusion = GaussianDiffusion(T=diffusion_steps, schedule="linear")
    pred_img, seg_seq = diffusion.ddim_inverse(
        net=model,
        start_t=diffusion_steps,
        steps=diffusion_steps,
        x=x_t,
        intv=[daysq[:, i].to(torch.float32) for i in range(4)],
        treat_cond=[treatments_q[:, i].to(torch.float32) for i in range(4)],
        i_tg=i_tg,
        device=device
    )
    
    # Process predictions
    seg_seq = torch.sigmoid(seg_seq)
    
    # Save predictions
    np.save(
        os.path.join(session_path, f'prediction-slice-{slice_idx:03d}.npy'),
        pred_img.cpu().numpy()
    )
    np.save(
        os.path.join(session_path, f'segmentation-slice-{slice_idx:03d}.npy'),
        seg_seq.cpu().numpy()
    )
    
    # Save visualizations
    save_visualization_results(
        session_path=session_path,
        file_prefix=f'target-sess-{target_idx:02d}-slice-{slice_idx:03d}',
        images={
            'prediction': torch.mean(pred_img, 0).cpu().numpy()
        },
        masks={
            'pred_mask': torch.mean(torch.sigmoid(seg_seq), 0).cpu().numpy()
        }
    )

def main():
    """Main execution function for TaDiff inference.
    example: python .\inference.py --patient_ids 17 --slice_idx 102 --input_day 10000 --input_treatment 1
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_ids', nargs='+', default=['17'],
                        help='List of patient IDs to process')
    parser.add_argument('--data_root', default='./data/sailor',
                        help='Root directory containing input data')
    parser.add_argument('--prefix', default='sub-',
                        help='Filename prefix for input files')
    parser.add_argument('--ckpt_path', default='./ckpt/model.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--diffusion_steps', type=int, default=50,
                        help='Number of diffusion steps')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of predictions to generate')
    parser.add_argument('--output_dir', default='./inference_results',
                        help='Directory to save results')
    parser.add_argument('--slice_idx', type=int, required=True,
                        help='Z-index of slice to process')
    parser.add_argument('--input_day', type=float, required=True,
                        help='Day value for prediction session')
    parser.add_argument('--input_treatment', type=float, required=True,
                        help='Treatment code for prediction session')
    args = parser.parse_args()
    
    # Setup device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Tadiff_model.load_from_checkpoint(args.ckpt_path, strict=False)
    model.eval()
    model.to(device)
    
    # Load data
    input_files = get_input_files(
        data_root=args.data_root,
        patient_ids=args.patient_ids,
        prefix=args.prefix
    )
    dataloader = load_data(input_files)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all data
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx}")
        
        # Process with input parameters
        patient_id = args.patient_ids[batch_idx]
        input_days = torch.tensor([args.input_day], device=device)
        input_treatments = torch.tensor([args.input_treatment], device=device)
        
        process_session(
            patient_id=patient_id,
            batch=batch,
            model=model,
            device=device,
            save_path=args.output_dir,
            input_days=input_days,
            input_treatments=input_treatments,
            slice_idx=args.slice_idx,
            diffusion_steps=args.diffusion_steps,
            num_samples=args.num_samples
        )

if __name__ == '__main__':
    main()