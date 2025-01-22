"""
TaDiff Model Testing Script
This script handles testing of the TaDiff model and generates visualizations of the results.
"""
from typing import List, Dict, Optional

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List

from src.tadiff_model import Tadiff_model
from src.net.diffusion import GaussianDiffusion
from src.data.data_loader import load_data
from src.visualization.visualizer import (
    plot_uncertainty_figure,
    save_visualization_results,
    create_directory
)
from src.evaluation.metrics import (
    setup_metrics,
    calculate_metrics,
    calculate_tumor_volumes,
    get_slice_indices
)
from src.utils.image_processing import (
    # normalize_to_255,
    to_pil_image,
    prepare_image_batch,
    create_noise_tensor,
    extract_slice
)

def process_session(
    patient_id: str,
    session_idx: int,
    batch: Dict[str, torch.Tensor],
    model: Tadiff_model,
    device: torch.device,
    metrics: Dict,
    save_path: str,
    diffusion_steps: int = 600,
    num_samples: int = 5,
    target_idx: int = 3
) -> Dict[str, Dict]:
    """
    Process a single session of data.
    
    Args:
        session_idx: Index of the session to process
        batch: Batch of data
        model: TaDiff model
        device: Device to run on
        metrics: Dictionary of metrics
        save_path: Path to save results
        diffusion_steps: Number of diffusion steps
        num_samples: Number of samples to generate
        target_idx: Target session index
        
    Returns:
        Dictionary of evaluation scores
    """
    session_scores = {}
    # patient_id = patient_id
    
    # Extract tensors from batch
    labels = batch['label'].to(device)
    images = batch['image'].to(device)
    days = batch['days'].to(device)
    treatments = batch['treatment'].to(device)
    

    
    # Calculate volumes and get important slices
    z_mask_size = calculate_tumor_volumes(labels[0, :, :, :, :])
    mean_vol = z_mask_size[z_mask_size > 0].mean()
    
    # Adjust minimum volume threshold
    n_sessions = labels.shape[1]
    if mean_vol < n_sessions * 100 and z_mask_size.max() > n_sessions * 200:
        mean_vol = n_sessions * 100
    z_mask_size[z_mask_size < mean_vol] = 0
    
    # b, cs, h, w, z = images.shape # b, c*sess, h, w, z, eg. [1, 59, 192, 192, 192]
    # images = images.view(b, 4, n_sessions, h, w, z) # t1, t1c, flair, t2
    # images = images.permute(0, 2, 1, 3, 4, 5) # b, s, c, h, w, z
    # images = images[:, :, :-1, :, :, :]  # remove T2 modal, b, s, c-1, h, w, z
    
    images = prepare_image_batch(images, n_sessions)
    
    # Get slice indices
    top_k_indices = get_slice_indices(z_mask_size, top_k=3)
    
    # Create session directory
    session_path = os.path.join(save_path, f'p-{patient_id}', f'-ses-{session_idx:02d}')
    create_directory(session_path)
    
    # Process each slice
    for slice_idx in top_k_indices:
        slice_scores = process_slice(
            slice_idx=slice_idx.item(),
            session_idx=session_idx,
            images=images,
            labels=labels,
            days=days,
            treatments=treatments,
            model=model,
            device=device,
            metrics=metrics,
            session_path=session_path,
            diffusion_steps=diffusion_steps,
            num_samples=num_samples,
            target_idx=target_idx
        )
        session_scores.update(slice_scores)
    
    return session_scores

def process_slice(
    slice_idx: int,
    session_idx: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    days: torch.Tensor,
    treatments: torch.Tensor,
    model: Tadiff_model,
    device: torch.device,
    metrics: Dict,
    session_path: str,
    diffusion_steps: int,
    num_samples: int,
    target_idx: int
) -> Dict[str, Dict]:
    """Process a single slice of data."""
    # Prepare data
    slice_indices = [slice_idx] * num_samples
    session_indices = np.array([
        session_idx - 3,
        session_idx - 2,
        session_idx - 1,
        session_idx
    ])
    session_indices[session_indices < 0] = 0
    session_indices = list(session_indices)
    # Extract relevant slices
    masks = labels[0, session_indices, :, :, :]
    masks = masks[:, :, :, slice_indices].permute(3, 0, 1, 2)
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
    print(f'x_0.shape: {x_0.shape}, x_t.shape: {x_t.shape}')
    # Prepare condition tensors
    daysq = days[0, session_indices].repeat(num_samples, 1)
    treatments_q = treatments[0, session_indices].repeat(num_samples, 1)
    
    # Run diffusion
    diffusion = GaussianDiffusion(T=diffusion_steps, 
                                  schedule="linear", # ,#"cosine", # 
                                  )
    # pred_img, seg_seq = diffusion.TaDiff_inverse(
    pred_img, seg_seq = diffusion.ddim_inverse(
    # pred_img, seg_seq = diffusion.dpm_solver_plus_plus_inverse(
        net=model,
        start_t=diffusion_steps,
        steps=diffusion_steps,
        x=x_t,
        intv=[daysq[:, i].to(torch.float32) for i in range(4)],
        treat_cond=[treatments_q[:, i].to(torch.float32) for i in range(4)],
        i_tg=i_tg,
        device=device
    )
    # pred_img, seg_seq = diffusion.unified_inverse_process(
    #     net=model,
    #     steps=diffusion_steps,
    #     x=x_t,
    #     intv=[daysq[:, i].to(torch.float32) for i in range(4)],
    #     treat_cond=[treatments_q[:, i].to(torch.float32) for i in range(4)],
    #     i_tg=i_tg,
    # )
    
    # Process predictions
    seg_seq = torch.sigmoid(seg_seq)
    predictions = {
        'images': pred_img, # (samples, C, H, W)  # C = 3 modality 
        'masks': seg_seq, # (samples, 4, H, W)
        'ground_truth': x_0, # (samples, C, H, W)
        'target_masks': masks # (samples, 4, H, W)
    }
    
    # Calculate metrics and save visualizations
    slice_scores = evaluate_predictions(
        predictions=predictions,
        metrics=metrics,
        session_idx=session_idx,
        slice_idx=slice_idx,
        session_path=session_path
    )
    
    return slice_scores

def evaluate_predictions(
    predictions: Dict[str, torch.Tensor],
    metrics: Dict,
    session_idx: int,
    slice_idx: int,
    session_path: str
) -> Dict[str, Dict]:
    """Evaluate predictions and save visualizations."""
    scores = {}
    
    # Calculate average predictions
    avg_img = torch.mean(predictions['images'], 0) # (3, H, W)  # 3 modality
    # avg_mask = torch.mean(predictions['masks'], 0)[0]
    avg_mask_pred = torch.sigmoid(predictions['masks'])
    avg_mask_pred = torch.mean(avg_mask_pred, 0) # (4, H, W)
    # predictions = {
    #     'images': pred_img, # (samples, C, H, W)  # C = 3 modality 
    #     'masks': seg_seq, # (samples, 4, H, W)
    #     'ground_truth': x_0, # (samples, C, H, W)
    #     'target_masks': masks # (samples, 4, H, W)
    # }
    # Save visualization results
    
    save_visualization_results(
        session_path=session_path,
        file_prefix=f'target-sess-{session_idx:02d}-slice-{slice_idx:03d}',
        images={
            'prediction': avg_img.cpu().numpy(),
            'ground_truth': predictions['ground_truth'][0].cpu().numpy()
        },
        masks={
            'pred_mask': avg_mask_pred.cpu().numpy(),
            'gt_mask': predictions['target_masks'][0].cpu().numpy(),
            'ref_mask': predictions['target_masks'][0].cpu().numpy(),
        }
    )
    
    # Calculate metrics for each sample
    for i in range(len(predictions['images'])):
        sample_metrics = calculate_metrics(
            metrics=metrics,
            prediction=predictions['images'][i].unsqueeze(0),
            ground_truth=predictions['ground_truth'][i].unsqueeze(0)
        )
        scores[f'sample_{i}'] = sample_metrics
    
    # Calculate metrics for ensemble prediction
    ensemble_metrics = calculate_metrics(
        metrics=metrics,
        prediction=avg_img.unsqueeze(0),
        ground_truth=predictions['ground_truth'][0].unsqueeze(0)
    )
    scores['ensemble'] = ensemble_metrics
    print(scores)
    return scores

def get_test_files(data_root: str = "./data/lumiere",
                   patient_ids: Optional[List[str]] = None,
                   prefix: str = '') -> List[Dict[str, str]]:
    """
    Get list of test files for specified patients.
    
    Args:
        data_root: Root directory containing the data
        patient_ids: List of patient IDs to process. If None, defaults to ['042']
        
    Returns:
        List of dictionaries containing file paths for each patient
    """
    # if patient_ids is None:
    #     patient_ids = ['042']
        
    npz_keys = ['image', 'label', 'days', 'treatment']
    test_files = []
    
    for patient_id in patient_ids:
        file_dict = {}
        for key in npz_keys:
            file_dict[key] = os.path.join(data_root, f'{prefix}{patient_id}_{key}.npy')
        test_files.append(file_dict)
        
    return test_files

def main():
    """Main execution function."""
    # Configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "./ckpt/s2_w_val_loss=0.00646-val_mse=0.0028-val_dice=0.811.ckpt"
    
    DIFFUSION_STEPS = 50
    NUM_SAMPLES = 4
    
    # Setup model and metrics
    model = Tadiff_model.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
    model.eval()
    model.to(DEVICE)
    
    metrics = setup_metrics(DEVICE)
    
    # Load data
    # patient_ids = ['042']
    patient_ids = ['17']
    test_files = get_test_files(data_root="./data/sailor", patient_ids=patient_ids,
                                prefix='sub-')
    dataloader = load_data(test_files)
    SAVE_PATH = './paper_sailor_eval_p17_ddim_step50'
    
    # Process all data
    all_scores = {}
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx}")
        
        # Process each session
        for target_session_idx in range(batch['label'].shape[1]):
            if target_session_idx > 0: 
                session_scores = process_session(
                    patient_id=patient_ids[batch_idx],
                    session_idx=target_session_idx,
                    batch=batch,
                    model=model,
                    device=DEVICE,
                    metrics=metrics,
                    save_path=SAVE_PATH,
                    diffusion_steps=DIFFUSION_STEPS,
                    num_samples=NUM_SAMPLES
                )
                all_scores.update(session_scores)
            else:
                continue
    
    # Save results
    results_file = f'test-score_diffusionsteps-{DIFFUSION_STEPS}_samples-{NUM_SAMPLES}.csv'
    pd.DataFrame.from_dict(all_scores, orient='index').to_csv(
        os.path.join(SAVE_PATH, results_file),
        float_format='%.3f'
    )

if __name__ == '__main__':
    main()
