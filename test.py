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
    normalize_to_255,
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
    if mean_vol < session_idx * 100 and z_mask_size.max() > session_idx * 200:
        mean_vol = session_idx * 100
    z_mask_size[z_mask_size < mean_vol] = 0
    
    # Get slice indices
    top_k_indices = get_slice_indices(z_mask_size, 3)
    
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
    
    # Extract relevant slices
    masks = labels[0, session_indices, :, :, :]
    masks = masks[:, :, :, slice_indices].permute(3, 0, 1, 2)
    print(images.shape)
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
    diffusion = GaussianDiffusion(T=diffusion_steps, schedule='linear')
    pred_img, seg_seq = diffusion.TaDiff_inverse(
        net=model,
        start_t=diffusion_steps // 1.5,
        steps=diffusion_steps // 1.5,
        x=x_t,
        intv=[daysq[:, i].to(torch.float32) for i in range(4)],
        treat_cond=[treatments_q[:, i].to(torch.float32) for i in range(4)],
        i_tg=i_tg,
        device=device
    )
    
    # Process predictions
    seg_seq = torch.sigmoid(seg_seq)
    predictions = {
        'images': pred_img,
        'masks': seg_seq,
        'ground_truth': x_0,
        'target_masks': masks
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
    avg_img = torch.mean(predictions['images'], 0)[0]
    # avg_mask = torch.mean(predictions['masks'], 0)[0]
    avg_mask = torch.sigmoid(predictions['masks'])[0]
    
    # Save visualization results
    save_visualization_results(
        session_path=session_path,
        file_prefix=f'target-sess-{session_idx:02d}-slice-{slice_idx:03d}',
        images={
            'prediction': avg_img.cpu().numpy(),
            'ground_truth': predictions['ground_truth'][0].cpu().numpy()
        },
        masks={
            'prediction': avg_mask.cpu().numpy(),
            'ground_truth': predictions['target_masks'][0].cpu().numpy()
        }
    )
    
    # Calculate metrics for each sample
    for i in range(len(predictions['images'])):
        sample_metrics = calculate_metrics(
            metrics=metrics,
            prediction=predictions['images'][i],
            ground_truth=predictions['ground_truth'][i]
        )
        scores[f'sample_{i}'] = sample_metrics
    
    # Calculate metrics for ensemble prediction
    ensemble_metrics = calculate_metrics(
        metrics=metrics,
        prediction=avg_img.unsqueeze(0),
        ground_truth=predictions['ground_truth'][0].unsqueeze(0)
    )
    scores['ensemble'] = ensemble_metrics
    
    return scores

def get_test_files(data_root: str = "./data/lumiere", patient_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
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
            file_dict[key] = os.path.join(data_root, f'{patient_id}_{key}.npy')
        test_files.append(file_dict)
        
    return test_files

def main():
    """Main execution function."""
    # Configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "./ckpt/s2_w_val_loss=0.00646-val_mse=0.0028-val_dice=0.811.ckpt"
    SAVE_PATH = './paper_lumiere_eval_p42_ft2'
    DIFFUSION_STEPS = 20
    NUM_SAMPLES = 3
    
    # Setup
    model = Tadiff_model.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
    model.eval()
    model.to(DEVICE)
    
    metrics = setup_metrics(DEVICE)
    
    patient_ids = ['042']
    test_files = get_test_files(data_root="./data/lumiere", patient_ids=patient_ids)
    print(test_files)
    dataloader = load_data(test_files)
    
    # Process all data
    all_scores = {}
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx}")
        
        imgs = batch['image'].to(DEVICE)
        b, cs, h, w, z = imgs.shape # b, c*sess, h, w, z, eg. [1, 59, 192, 192, 192]
        imgs = imgs.view(b, 4, batch['label'].shape[1], h, w, z) # t1, t1c, flair, t2
        imgs = imgs.permute(0, 2, 1, 3, 4, 5) # b, s, c, h, w, z
        imgs = imgs[:, :, :-1, :, :, :]  # remove T2 modal, b, s, c-1, h, w, z
        batch['image'] = imgs
        
        for session_idx in range(batch['label'].shape[1]):
            session_scores = process_session(
                patient_id = patient_ids[batch_idx],
                session_idx=session_idx,
                batch=batch,
                model=model,
                device=DEVICE,
                metrics=metrics,
                save_path=SAVE_PATH,
                diffusion_steps=DIFFUSION_STEPS,
                num_samples=NUM_SAMPLES
            )
            all_scores.update(session_scores)
    
    # Save results
    results_file = f'test-score_diffusionsteps-{DIFFUSION_STEPS}_samples-{NUM_SAMPLES}.csv'
    pd.DataFrame.from_dict(all_scores, orient='index').to_csv(
        os.path.join(SAVE_PATH, results_file),
        float_format='%.3f'
    )

if __name__ == '__main__':
    main()

# if __name_
# mae = MAE().to(device)
# psnr = PSNR().to(device)
# # ssim = SSIM(data_range=255).to(device)
# ssim = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3).to(device)


# def ravd(pred, truth, threshold=0.5):
#     """
#     relative absoluted volume difference.
#     """
#     pred = pred.cpu().detach().numpy()
#     truth = truth.cpu().detach().numpy()
#     pred = (pred>=threshold).astype(bool)
#     truth = truth.astype(bool)
#     vol1 = np.count_nonzero(pred)+1
#     vol2 = np.count_nonzero(truth)+1
    
#     return (vol1-vol2)/float(vol2), vol1-1


# scores = dict()

# for i, batch in enumerate(dataloader):
#     Patient_id = p_ids[i]
#     print(f'patient-{i}, id={p_ids[i]}')
#     # npz_keys = ['image', 'label', 'days', 'treatment']
    
#     labels = batch['label'].to(device)
#     imgs = batch['image'].to(device)
#     days = batch['days'].to(device) # b, s, -> (s,)
#     treats = batch['treatment'].to(device) # b, s -> (s)
    
#     s = labels.shape[1]  # eg. [1, 14, 192, 192, 192]
#     # cs = 4*s
#     b, cs, h, w, z = imgs.shape # b, c*sess, h, w, z, eg. [1, 59, 192, 192, 192]
#     imgs = imgs.view(b, 4, s, h, w, z) # t1, t1c, flair, t2
#     imgs = imgs.permute(0, 2, 1, 3, 4, 5) # b, s, c, h, w, z
#     imgs = imgs[:, :, :-1, :, :, :]  # remove T2 modal, b, s, c-1, h, w, z
    
    
#     z_mask_size = torch.sum(labels[0, :, :, :, :], axis=(0, 1, 2)).float() #/ s # 192, 
#     # n_nonzero = torch.nonzero(z_mask_size).size(0)
#     mean_vol = z_mask_size[z_mask_size>0].mean()
#     if mean_vol < s*100 and z_mask_size.max() > s*200:
#         mean_vol = s*100. # threshold minimum cm 5cmxcm
#     z_mask_size[z_mask_size<mean_vol] = 0
    
#     vol_all = torch.sum(labels[0, :, :, :, :], axis=(1, 2, 3)).cpu().detach().numpy() # s, 

#     _, topk_zi = torch.topk(z_mask_size, Top_K)
#     topk_zi = list(topk_zi.squeeze().cpu().detach().numpy())
    
#     # z_mask_size[z_mask_size < Min_v] = 0
#     nonzero_zi = list(torch.nonzero(z_mask_size).squeeze().cpu().detach().numpy())
    
#     print(f'total nonzero {len(nonzero_zi)}')
    
#     col_pre = {'patient': p_ids[i], 
#                'sess': -1,
#                'day': 0, 
#                'delta_day': 0,
#                'treat': 1,
#                'ml': 0,
#                'ml-ref': 0,}
#     # # ---------- save tumor information to csv --------------------------  comment out -------- 
#     # non_label = labels[0, :, :, :, topk_zi]
#     # tumor_dict = dict()
#     # seq_tumor_volume = torch.sum(labels[0, :, :, :, :], axis=(1, 2, 3)) #  (sess, )
    
#     # tumor_dict['treats'] = treats[0].numpy()
#     # tumor_dict['days'] = days[0].numpy()
#     # tumor_dict['volume'] = seq_tumor_volume.numpy() / 1000.  # ml
    
#     # zi_seq_tumor_size = torch.sum(non_label, axis=(1, 2))
#     # for j, zi in enumerate(topk_zi):
#     #     tumor_dict[f'{zi}-size'] = zi_seq_tumor_size[:, j].numpy() / 100.
    
#     # pd.DataFrame.from_dict(data=tumor_dict, orient='index').to_csv(f'{p_ids[i]}_dict_file.csv', header=False)
#     # continue
    
#     # --------------- prepare S U f, subset of all sessions , to evaluate ------------------------------

#     # slicing z axial, len N_sp is the soft-batch size, aslo the numble of sampling
    
#     for tg_sess in range(s):
        
#         col_pre['sess'] = f'{tg_sess:02d}'
#         col_pre['day'] = days[0][tg_sess].cpu().detach().numpy()
#         col_pre['delta_day'] = 0. if tg_sess < 1 else (days[0][tg_sess] - days[0][tg_sess-1]).cpu().detach().numpy()
#         col_pre['treat'] = treats[0][tg_sess].cpu().detach().numpy() #- 1 test 
#         col_pre['ml'] = np.round(vol_all[tg_sess]/1000., 3)
#         if tg_sess < 1:
#             col_pre['ml-ref'] = np.round(vol_all[tg_sess]/1000., 3)
#             # continue # start from session 1
#         else:
#             col_pre['ml-ref'] = np.round(vol_all[tg_sess-1]/1000., 3)
        
#         patient_path = os.path.join(save_path, f'p-{Patient_id}')
#         mkpath(patient_path)
#         sess_path = os.path.join(patient_path, f'-ses-{tg_sess:02d}')
#         mkpath(sess_path)
        
#         # if tg_sess > 6:  # ignore rest, for debug, should comment out after debugging
#         #     continue
#         # topk_zi = [65, ] #91, 101, 110]
#         if tg_sess == 0: 
#             end_idx = 1
#         else:
#             end_idx = len(nonzero_zi) if Top_K > len(nonzero_zi) else Top_K
            
#         print(f'----only inference to slice ----- {end_idx} of all {len(nonzero_zi)}----')
        
#         for k, z_idx in enumerate(topk_zi[:end_idx]): # enumerate(nonzero_zi)
#             # n_k = k
#             if (k+1)%2:  # skip 1, 3, 5, 7 ... slices
#                 n_k = (k+1)//2
#             else:
#                 continue
#         # for k, z_idx in enumerate(nonzero_zi): # enumerate(nonzero_zi)
            
#         # for k, z_idx in enumerate(topk_zi[:1]): # just infer on the top1 slice per session for debugging
#             zi = [z_idx] * N_sp

#             # select s1, s2, s3, f
#             Sf = np.array([tg_sess-3, tg_sess-2, tg_sess-1, tg_sess])
#             # Sf = np.array([tg_sess-1, tg_sess-1, tg_sess-1, tg_sess])
#             # Sf = np.array([tg_sess-2, tg_sess-1, tg_sess-1, tg_sess])
#             Sf[Sf<0] = 0
#             Sf = list(Sf)
                
#             # Sf = [0, 1, 2, 3]  # indice of session 
            
#             i_tg = F_idx * torch.ones((len(zi),), dtype=torch.int8).to(device) # target sess idex, -1: the last, [0, 1, 2, -1]    

#             sb = len(zi)
#             masks = labels[0, Sf, :, :, :]
#             # if torch.sum(labels[0,-1, :, :]) < 10:
#             #     continue
#             masks = masks[:, :, :, zi].permute(3,0,1,2)                  # zi, sf, h, w
#             seq_imgs = imgs[0, Sf, :, :, :, :]
#             seq_imgs = seq_imgs[:, :, :, :, zi].permute(4, 0, 1, 2, 3) # zi, sf, c, h, w
            
            
#             noise = torch.randn((sb, 3, h, w)) # x_0.shape, sb, c, h, w
#             x_0 = []
#             x_t = seq_imgs.clone()  # zi, sf, c, h, w
            
            
#             # if Sf[-1] == Sf[-2]:
#             #     x_t[:, :-1, :, :, :] = 0.
#                 # masks[:, :-1, :, :] = 0
                
#             for i, j in zip(range(sb), i_tg):
#                 x_0 += [seq_imgs[[i], j, :, :, :]]
#                 x_t[i, j, :, :, :] = noise[i, :, :, :]  # replace target x_0 with noise
#             x_0 = torch.cat(x_0, dim=0)  # zi, c, h, w
            
#             x_t = x_t.reshape(sb, len(Sf)*3, h, w)
        
#             daysq = days[0, Sf].repeat(sb, 1)                            # zi, sf
#             treatments = treats[0, Sf].repeat(sb, 1)                     # zi, sf
            

#             intvs = [daysq[:, i].to(torch.float32) for i in range(4)]
#             treat_cond = [treatments[:, i].to(torch.float32) for i in range(4)]


#             # --- initial diffusion ----------------------------------------------------
#             diffusion = GaussianDiffusion(T=T, schedule='linear')
#             # noise = torch.randn((x_0.shape))
#             # x_t = torch.cat([img_cond, noise], dim=1)
#             pred_img, seg_seq = diffusion.TaDiff_inverse(model, start_t=T//1.5, steps=T//1.5, 
#                                                         x=x_t.to(device), intv=intvs,  
#                                                         treat_cond=treat_cond, i_tg=i_tg,
#                                                         device=device,
#                                                         )
#             # seg_seq : zi, sf, h, w  === mask
#             # x_0  =========== pred_img          zi, 3(t1,t1c,flair), h, w
            
            
#             avg_img = torch.mean(pred_img, 0) # 3, h, w,            x_0[0]
#             img_std = torch.std(pred_img, 0) # 3, h, w              hotmap, t1, t1c, flair
            
#             # pred_img = clam_255(pred_img)
#             # x_0 = clam_255(x_0)
#             # avg_img = clam_255(avg_img)
            
#             seg_seq = torch.sigmoid(seg_seq)
#             # seg_seq = (seg_seq > 0.5) * 1 
                
#             mask_p = torch.cat([seg_seq[[i], j, :, :] for i, j in zip(range(sb), i_tg)], 0)
#             mask_gt = torch.cat([masks[[i], j, :, :] for i, j in zip(range(sb), i_tg)], 0) * 1 
#             mask_rf = torch.cat([masks[[i], j-1, :, :] for i, j in zip(range(sb), i_tg)], 0) * 1 
#             mask_p_rf = torch.cat([seg_seq[[i], j-1, :, :] for i, j in zip(range(sb), i_tg)], 0)

#             avg_mask_p_rf = torch.mean(mask_p_rf,  0) # h, w        mask_rf[0]
#             avg_mask = torch.mean(mask_p,  0) # h, w                mask_gt[0]
#             mask_std = torch.std(mask_p, 0) # h, w                  hotmap
#             seg_seq_std = torch.std(seg_seq, 0) # h, w                  hotmap
            
#             avg_all_masks = torch.mean(seg_seq, 0)
            
#             # csv metrics collum: volume, RAVD, DICE, SSIM, MAE, PSNR, DICE25, RAVD25, DICE75, RAVD75
#             # index: ground-truth, sample1, sample2, sample3, sample4, sample5, ensemble, uncertianty
            
#             vol_gt = torch.sum(mask_gt[0, :, :]).cpu().detach().numpy()
#             vol_rf = torch.sum(mask_rf[0, :, :]).cpu().detach().numpy()
            
#             col_z = {
#                 'z-slice': f'{n_k:02d}-{z_idx:03d}',
#                 'gt-vol': np.round(vol_gt/100., 3),
#                 'ref-vol': np.round(vol_rf/100., 3), 
#             }
            
#             col_pre.update(col_z.copy())
#             # col = {'Vol-r': np.round(vol_rf/100., 3), 'Vol': np.round(vol_gt/100., 3),
#             #        'RAVD': 0,  'DICE': 1., 'SSIM': 1.,   "MAE": 0.,  'PSNR': 1, 
#             #        'DICE25': 1,  'RAVD25': 0,  'DICE75': 1,  'RAVD75': 0, 'DICE-rf': 1, 'RAVD-rf': 0,
#             #        }
            
#             # scores['Ground-Truth'] = col

#             # all_scores[f'{patient_id}_sess_{t_sess_idx}__zslice_{z_idx}'] = slice_scores
            
#             for i in range(sb+1):
#                 if i < sb:
#                     ssim_score = ssim(clam_255(pred_img[[i]]), clam_255(x_0[[i]])).cpu().detach().numpy()
#                     mae_score = mae(pred_img[[i]].clamp(-1, 1), x_0[[i]].clamp(-1, 1)).cpu().detach().numpy()
#                     psnr_score = psnr(clam_255(pred_img[[i]]), clam_255(x_0[[i]])).cpu().detach().numpy()
#                     dice25 = dice_th025(mask_p[[i]], mask_gt[[i]]).cpu().detach().numpy()
#                     dice50 = dice_th05(mask_p[[i]], mask_gt[[i]]).cpu().detach().numpy()
#                     dice75 = dice_th075(mask_p[[i]], mask_gt[[i]]).cpu().detach().numpy()
#                     ravd25,_ = ravd(mask_p[i], mask_gt[i], threshold=0.25)
#                     ravd50, vol = ravd(mask_p[i], mask_gt[i], threshold=0.50)
#                     ravd75, vol_pr = ravd(mask_p[i], mask_gt[i], threshold=0.75)
#                     ravd_rf, _ = ravd(mask_p_rf[i], mask_rf[i], threshold=0.50)
#                     dice_rf = dice_th05(mask_p_rf[i], mask_rf[[i]]).cpu().detach().numpy()
                    
#                     col = {'infer': i, 'pred-vol': np.round(vol/100., 3), 'pred-vol-r': np.round(vol_pr/100., 3), 
#                            'RVD': np.round(ravd50,3),  'DICE': np.round(dice50, 3),  'SSIM': np.round(ssim_score, 3), 
#                            "MAE": np.round(mae_score, 3), 'PSNR': np.round(psnr_score, 3), 
#                            'DICE25': np.round(dice25, 3), 'RAVD25': np.round(ravd25,3), 
#                            'DICE75': np.round(dice75,3), 'RAVD75': np.round(ravd75, 3), 
#                            'DICE-rf': np.round(dice_rf,3), 'RAVD-rf': np.round(ravd_rf,3),
#                            }
#                     idx_name = f'{Patient_id}-s-{tg_sess:02d}-z-{n_k:02d}-{z_idx:03d}-inf-{i}'
#                     col_pre.update(col.copy())
#                     scores[idx_name] = col_pre.copy()
#                     # scores[f'Sample{i}'] = col
#                     print(col)
#                 else: # compute ensemble scores
#                     ssim_score = ssim(clam_255(avg_img[None, ...]), clam_255(x_0[[0]])).cpu().detach().numpy()
#                     mae_score = mae(avg_img[None, ...].clamp(-1, 1), x_0[[0]].clamp(-1, 1)).cpu().detach().numpy()
#                     psnr_score = psnr(clam_255(avg_img[None, ...]), clam_255(x_0[[0]])).cpu().detach().numpy()
#                     dice25 = dice_th025(avg_mask[None, ...], mask_gt[[0]]).cpu().detach().numpy()
#                     dice50 = dice_th05(avg_mask[None, ...], mask_gt[[0]]).cpu().detach().numpy()
#                     dice75 = dice_th075(avg_mask[None, ...], mask_gt[[0]]).cpu().detach().numpy()
#                     ravd25,_ = ravd(avg_mask, mask_gt[0], threshold=0.25)
#                     ravd50, vol = ravd(avg_mask, mask_gt[0], threshold=0.50)
#                     ravd75,_ = ravd(avg_mask, mask_gt[0], threshold=0.75)
#                     ravd_rf, _ = ravd(avg_mask_p_rf, mask_rf[0], threshold=0.50)
#                     dice_rf = dice_th05(avg_mask_p_rf[None, ...], mask_rf[[0]]).cpu().detach().numpy()
                    
#                     col = {'infer': sb, 'pred-vol': np.round(vol/100., 3), 
#                            'RVD': np.round(ravd50,3),  'DICE': np.round(dice50, 3),  'SSIM': np.round(ssim_score, 3), 
#                            "MAE": np.round(mae_score, 3), 'PSNR': np.round(psnr_score, 3), 
#                            'DICE25': np.round(dice25, 3), 'RAVD25': np.round(ravd25,3), 
#                            'DICE75': np.round(dice75,3), 'RAVD75': np.round(ravd75, 3),
#                            'DICE-rf': np.round(dice_rf,3), 'RAVD-rf': np.round(ravd_rf,3),
#                            }
                    
#                     idx_name = f'{Patient_id}-s-{tg_sess:02d}-z-{n_k:02d}-{z_idx:03d}-inf-{sb}'
#                     col_pre.update(col.copy())
#                     scores[idx_name] = col_pre.copy()
                    
#                     print(col)
                    
            
#             print('-------------save image-----------------')
#             file_prefix = f'{Patient_id}-targe-sess-{tg_sess:02d}-z-idx-{n_k:02d}-{z_idx:03d}'
#             avg_all_masks = avg_all_masks.cpu().detach().numpy()
#             avg_mask = avg_mask.cpu().detach().numpy()
#             avg_img = avg_img.cpu().detach().numpy()
#             img_std = img_std.cpu().detach().numpy()
#             mask_std = mask_std.cpu().detach().numpy()
#             mask_p = mask_p.cpu().detach().numpy()
#             mask_gt = mask_gt.cpu().detach().numpy()
#             pred_img = pred_img.cpu().detach().numpy()
#             x_0 = x_0.cpu().detach().numpy()
#             seg_seq_std = seg_seq_std.cpu().detach().numpy()
#             # mask_th05 = 
            
#             avg_mask_p_rf = avg_mask_p_rf.cpu().detach().numpy()
#             mask_rf = mask_rf.cpu().detach().numpy()
            
#             rf_mask = to_pil(mask_rf[0, :, :])
#             gt_mask = to_pil(mask_gt[0, :, :])
#             gt_mask.save(os.path.join(sess_path, f"{file_prefix}-gt-mask.png"))
            
#             # (avg_mask_p_rf>0.49).astype(np.uint8)
#             p_rf_mask = to_pil((avg_mask_p_rf>0.49).astype(np.uint8))
#             # rf_mask = p_rf_mask # only for lumere dataset, there is no groundtruth rf_mask. 
#             p_avg_mask = to_pil(avg_mask)
#             p_avg_mask.save(os.path.join(sess_path, f"{file_prefix}-ensemble-mask.png"))
#             for j in range(3):
#                 gt_img = to_pil(x_0[0, j, :, :])
#                 p_img = to_pil(avg_img[j, :, :])
#                 p_img.save(os.path.join(sess_path, f"{file_prefix}-ensemble-T{j}.png"))
#                 gt_img.save(os.path.join(sess_path, f"{file_prefix}-gt-T{j}.png"))
                
                
#                 gt_img = overlay_maps(gt_img, gt_mask, rf_mask)
#                 p_img =  overlay_maps(p_img, p_avg_mask, rf_mask)
                
#                 p_img.save(os.path.join(sess_path, f"{file_prefix}-ensemble-T{j}_overlay.png"))
#                 gt_img.save(os.path.join(sess_path, f"{file_prefix}-gt-T{j}_overlay.png"))
                
                
#                 gt_img = drawContour(gt_img, gt_mask, RGB=(255,0,0)) # red
#                 p_img = drawContour(p_img, p_avg_mask, RGB=(255,0,0)) # red
#                 # p_img = drawContour(p_img, rf_mask) # yellow
#                 p_img.save(os.path.join(sess_path, f"{file_prefix}-ensemble-T{j}_contour.png"))
#                 gt_img.save(os.path.join(sess_path, f"{file_prefix}-gt-T{j}_contour.png"))

                
#                 # if j == 1:
#                 #     gt_img = overlay_maps(gt_img, gt_mask, rf_mask)
#                 #     p_img =  overlay_maps(p_img, p_avg_mask, rf_mask)
#                 # elif j == 0:
#                 #     gt_img = overlay_maps(gt_img, gt_mask, gt_mask)
#                 #     p_img =  overlay_maps(p_img, p_avg_mask, p_avg_mask)
#                 # else:
#                 # if j == 2:
#                 #     gt_img = drawContour(gt_img, gt_mask, RGB=(255,0,0)) # red
#                 #     gt_img = overlay_maps(gt_img, gt_mask, rf_mask)
#                 #     # gt_img = drawContour(gt_img, rf_mask) # yellow
#                 #     p_img = drawContour(p_img, p_avg_mask, RGB=(255,0,0)) # red
#                 #     p_img =  overlay_maps(p_img, p_avg_mask, rf_mask)
                    
                    
#                 #     p_img.save(os.path.join(sess_path, f"{file_prefix}-ensemble-T{j}_overlay.png"))
#                 #     gt_img.save(os.path.join(sess_path, f"{file_prefix}-gt-T{j}_overlay.png"))
#                 # else:
#                 #     gt_img = drawContour(gt_img, gt_mask, RGB=(255,0,0)) # red
#                 #     p_img = drawContour(p_img, p_avg_mask, RGB=(255,0,0)) # red
#                 #     # p_img = drawContour(p_img, rf_mask) # yellow
#                 #     p_img.save(os.path.join(sess_path, f"{file_prefix}-ensemble-T{j}_contour.png"))
#                 #     gt_img.save(os.path.join(sess_path, f"{file_prefix}-gt-T{j}_contour.png"))
                    
                
#             # plot figure  for visualization analysis png
#             for i in range(sb+1):
#                 if i == 0: # save preded avg masks on reference sess 1, 2, 3 imgs
#                     arr = to_pil(avg_all_masks[0, :, :])
#                     arr.save(os.path.join(sess_path, f"{file_prefix}-segment-s1.png"))
#                     arr = to_pil(avg_all_masks[1, :, :])
#                     arr.save(os.path.join(sess_path, f"{file_prefix}-segment-s2.png"))
#                     arr = to_pil(avg_all_masks[2, :, :])
#                     arr.save(os.path.join(sess_path, f"{file_prefix}-segment-s3.png"))
#                 if i < sb:                                      # save pred masks
#                     arr = to_pil(mask_p[i, :, :])
#                     arr.save(os.path.join(sess_path, f"{file_prefix}-sample{i}-mask.png"))
#                     for j in range(3):                         # save pred images
#                         arr = to_pil(pred_img[i, j, :, :])
#                         arr.save(os.path.join(sess_path, f"{file_prefix}-sample{i}-T{j}.png"))

#                 else:                                                 # plot uncertianty maps
#                     plot_uncertainty_fig(mask_std, os.path.join(sess_path, f"{file_prefix}-uncertianty-mask.png"), overlay=avg_img[2, :, :])
#                     plot_uncertainty_fig(seg_seq_std[0], os.path.join(sess_path, f"{file_prefix}-segment-s1-uncert.png"))
#                     plot_uncertainty_fig(seg_seq_std[1], os.path.join(sess_path, f"{file_prefix}-segment-s2-uncert.png"))
#                     plot_uncertainty_fig(seg_seq_std[2], os.path.join(sess_path, f"{file_prefix}-segment-s3-uncert.png"))
                    
#                     # arr = to_pil(mask_std)
#                     # arr = Image.fromarray(arr)
#                     # arr.save(os.path.join(sess_path, f"{file_prefix}-uncertianty-mask.png"))
#                     for j in range(3):
#                         plot_uncertainty_fig(img_std[j, :, :], os.path.join(sess_path, f"{file_prefix}-uncertianty-T{j}.png"))
#                         # arr = to_pil(img_std[j, :, :])
#                         # arr = Image.fromarray(arr)
#                         # arr.save(os.path.join(sess_path, f"{file_prefix}-uncertianty-T{j}.png"))

# # file_prefix = 
# csv_file_name = f'test-score_diffusionstep-{T}_tok-{Top_K}_lumiere_37_cases.csv'

# pd.DataFrame.from_dict(scores.copy(),  orient='index').to_csv(os.path.join(save_path, csv_file_name), float_format='%.3f')                    
