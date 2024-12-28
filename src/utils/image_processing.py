"""
Image processing utilities for TaDiff model.
Handles image conversions, normalization, and basic image operations.
"""
from typing import Union, Optional, Tuple

import torch
import numpy as np
from PIL import Image

def to_binary(
    arr: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Convert array to binary mask using threshold.
    
    Args:
        arr: Input array/tensor
        threshold: Threshold value for binarization
        
    Returns:
        Binary mask as numpy array
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()
    
    th = arr.max() * threshold
    return (arr >= th).astype(np.uint8)

def normalize_tensor(
    tensor: torch.Tensor,
    dim: Tuple[int, ...] = (2, 3),
    keepdim: bool = True
) -> torch.Tensor:
    """
    Normalize tensor to range [0, 1].
    
    Args:
        tensor: Input tensor
        dim: Dimensions to compute min/max over
        keepdim: Whether to keep dimensions
        
    Returns:
        Normalized tensor
    """
    min_val = torch.amin(tensor, dim, keepdim=keepdim)
    max_val = torch.amax(tensor, dim, keepdim=keepdim)
    return (tensor - min_val) / (max_val - min_val + 1e-7)

def normalize_to_255(
    tensor: torch.Tensor,
    clamp_range: Tuple[float, float] = (-1, 1)
) -> torch.Tensor:
    """
    Normalize tensor to range [0, 255].
    
    Args:
        tensor: Input tensor
        clamp_range: Range to clamp values to before normalization
        
    Returns:
        Normalized tensor as uint8
    """
    return ((tensor.clamp(*clamp_range) - clamp_range[0]) * (255.0 / (clamp_range[1] - clamp_range[0]))).type(torch.uint8)

def to_pil_image(
    arr: Union[np.ndarray, torch.Tensor],
    to_rgb: bool = False
) -> Image.Image:
    """
    Convert array/tensor to PIL Image.
    
    Args:
        arr: Input array/tensor
        to_rgb: Whether to convert to RGB mode
        
    Returns:
        PIL Image
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()
    
    # Normalize to [0, 255]
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.9).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(arr)
    if to_rgb:
        img = img.convert('RGB')
    
    return img

def prepare_image_batch(
    images: torch.Tensor,
    num_sessions: int,
    modalities: int = 4
) -> torch.Tensor:
    """
    Prepare image batch for model input.
    
    Args:
        images: Input image tensor
        num_sessions: Number of sessions
        modalities: Number of modalities
        
    Returns:
        Prepared image tensor
    """
    batch, channels, height, width, depth = images.shape
    
    # Reshape to separate sessions and modalities
    images = images.view(batch, modalities, num_sessions, height, width, depth)
    
    # Permute to get sessions first
    images = images.permute(0, 2, 1, 3, 4, 5)
    
    # Remove last modality (typically T2)
    images = images[:, :, :-1, :, :, :]
    
    return images

def create_noise_tensor(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create random noise tensor.
    
    Args:
        batch_size: Batch size
        channels: Number of channels
        height: Height of tensor
        width: Width of tensor
        device: Device to create tensor on
        
    Returns:
        Noise tensor
    """
    noise = torch.randn((batch_size, channels, height, width))
    if device is not None:
        noise = noise.to(device)
    return noise

def extract_slice(
    volume: torch.Tensor,
    slice_indices: Union[torch.Tensor, list],
    axis: int = -1
) -> torch.Tensor:
    """
    Extract slices from volume.
    
    Args:
        volume: Input volume tensor
        slice_indices: Indices of slices to extract
        axis: Axis to extract slices from
        
    Returns:
        Extracted slices
    """
    if isinstance(slice_indices, list):
        slice_indices = torch.tensor(slice_indices, device=volume.device)
    
    # Move slice dimension to the end if not already there
    if axis != -1:
        dims = list(range(volume.dim()))
        dims.remove(axis)
        dims.append(axis)
        volume = volume.permute(*dims)
    
    return volume[..., slice_indices]

def merge_batch_slices(
    slices: torch.Tensor,
    indices: torch.Tensor,
    target_size: int
) -> torch.Tensor:
    """
    Merge batch of slices back into volume.
    
    Args:
        slices: Batch of slices
        indices: Original indices of slices
        target_size: Size of target dimension
        
    Returns:
        Merged volume
    """
    device = slices.device
    batch_size = slices.shape[0]
    
    # Initialize output tensor
    output_shape = list(slices.shape[1:])
    output_shape.append(target_size)
    output = torch.zeros(output_shape, device=device)
    
    # Place slices back in their original positions
    for i in range(batch_size):
        output[..., indices[i]] = slices[i]
    
    return output

# def normalize_to_1(tensor_arr):
#     """Normalize tensor to [0,1] range."""
#     min_arr = torch.amin(tensor_arr, (2,3), keepdim=True)
#     max_arr = torch.amax(tensor_arr, (2,3), keepdim=True)
#     return (tensor_arr-min_arr) / (max_arr - min_arr + 1.e-7)

# def to_pil_image(arr, to_rgb=False):
#     """Convert array to PIL Image."""
#     normalized = ((arr - arr.min()) / (arr.max() - arr.min() + 1.e-8) * 255.9).astype(np.uint8)
#     if to_rgb:
#         return Image.fromarray(normalized).convert('RGB')
#     return Image.fromarray(normalized)

# def create_noise_tensor(batch_size, channels, height, width, device):
#     """Create random noise tensor."""
#     return torch.randn((batch_size, channels, height, width)).to(device)

# def extract_slice(tensor, indices):
#     """Extract slice from tensor at given indices."""
#     return tensor[..., indices]
