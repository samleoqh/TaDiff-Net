# """Visualization utilities for model predictions."""
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageFilter
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# def create_directory(path):
#     """Create directory if it doesn't exist."""
#     if not os.path.exists(path):
#         os.makedirs(path)

# def to_pil_image(
#     arr: Union[np.ndarray, torch.Tensor],
#     to_rgb: bool = False
# ) -> Image.Image:
#     """
#     Convert array/tensor to PIL Image.
    
#     Args:
#         arr: Input array/tensor
#         to_rgb: Whether to convert to RGB mode
        
#     Returns:
#         PIL Image
#     """
#     if isinstance(arr, torch.Tensor):
#         arr = arr.cpu().detach().numpy()
    
#     # Normalize to [0, 255]
#     arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.9).astype(np.uint8)
    
#     # Convert to PIL Image
#     img = Image.fromarray(arr)
#     if to_rgb:
#         img = img.convert('RGB')
    
#     return img


# def plot_uncertainty_figure(arr, path_file='', overlay=None):
#     """Plot uncertainty heatmap."""
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     cx = np.arange(arr.shape[1])
#     cy = np.arange(arr.shape[0])
#     X, Y = np.meshgrid(cx, cy)
    
#     ax.set_aspect('equal')
#     col1 = ax.pcolormesh(X, Y, arr, cmap='magma',
#                         vmin=arr.min(), vmax=arr.max())
                        
#     if overlay is not None:
#         ax.pcolormesh(X, Y, overlay, cmap='gray', alpha=0.35)
        
#     ax.axis([cx.min(), cx.max(), cy.max(), cy.min()])
#     divider1 = make_axes_locatable(ax)
#     cax1 = divider1.append_axes("right", size="3%", pad=0.06)
#     cbar = fig.colorbar(col1, cax=cax1, extend='max')
#     cbar.ax.tick_params(labelsize=18)
    
#     plt.tight_layout()
#     plt.savefig(path_file, dpi=300, facecolor='black')
#     plt.close()

# def save_visualization_results(session_path, file_prefix, images, masks):
#     """Save visualization results including overlays and contours."""
#     # Save ground truth mask
#     gt_mask_pil = to_pil_image(masks['ground_truth'])
#     gt_mask_pil.save(os.path.join(session_path, f"{file_prefix}-gt-mask.png"))
    
#     # Save prediction mask
#     pred_mask_pil = to_pil_image(masks['prediction'])
#     pred_mask_pil.save(os.path.join(session_path, f"{file_prefix}-ensemble-mask.png"))
    
#     # Save images with overlays and contours
#     for j in range(3):  # For each modality
#         gt_img = to_pil_image(images['ground_truth'][j])
#         pred_img = to_pil_image(images['prediction'][j])
        
#         # Save original images
#         pred_img.save(os.path.join(session_path, f"{file_prefix}-ensemble-T{j}.png"))
#         gt_img.save(os.path.join(session_path, f"{file_prefix}-gt-T{j}.png"))
        
#         # Save images with overlays
#         gt_overlay = create_overlay(gt_img, gt_mask_pil)
#         pred_overlay = create_overlay(pred_img, pred_mask_pil)
        
#         gt_overlay.save(os.path.join(session_path, f"{file_prefix}-gt-T{j}_overlay.png"))
#         pred_overlay.save(os.path.join(session_path, f"{file_prefix}-ensemble-T{j}_overlay.png"))
        
#         # Save images with contours
#         gt_contour = draw_contour(gt_img, gt_mask_pil)
#         pred_contour = draw_contour(pred_img, pred_mask_pil)
        
#         gt_contour.save(os.path.join(session_path, f"{file_prefix}-gt-T{j}_contour.png"))
#         pred_contour.save(os.path.join(session_path, f"{file_prefix}-ensemble-T{j}_contour.png"))

# def create_overlay(image, mask, transparency=0.4):
#     """Create overlay of mask on image."""
#     # Implementation details...
#     pass

# def draw_contour(image, mask, color=(255,0,0)):
#     """Draw contour of mask on image."""
#     # Implementation details...
#     pass




"""
Visualization utilities for TaDiff model results.
Handles plotting, image saving, and visualization of uncertainty maps.
"""
import os
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image, ImageFilter

def to_pil(arr, toRGB=False):
    if toRGB:
        return Image.fromarray((((arr - arr.min()) / (arr.max() - arr.min() + 1.e-8)) * 255.9).astype(np.uint8)).convert('RGB')
    else:
        return Image.fromarray((((arr - arr.min()) / (arr.max() - arr.min() + 1.e-8)) * 255.9).astype(np.uint8))


# Configure matplotlib settings
mpl.rc('image', cmap='gray')
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white"
})

# Color mapping for different classes
COLOR_MAP = {
    0: (0, 0, 0),      # background, black
    1: (0, 255, 0),    # class 1, green/growth
    2: (0, 0, 255),    # class 2, blue/shrinkage
    3: (255, 0, 0),    # class 3, red/stable tumor
}

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def plot_uncertainty_figure(
    arr: np.ndarray,
    save_path: str,
    overlay: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> None:
    """
    Plot and save uncertainty figure.
    
    Args:
        arr: Array containing uncertainty values
        save_path: Path to save the figure
        overlay: Optional overlay image
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize, sharex=False, sharey=False)
    
    cx = np.arange(arr.shape[1])
    cy = np.arange(arr.shape[0])
    X, Y = np.meshgrid(cx, cy)
    
    ax.set_aspect('equal')
    col1 = ax.pcolormesh(X, Y, arr, cmap='magma',
                        vmin=arr.min(), vmax=arr.max())
                        
    if overlay is not None:
        ax.pcolormesh(X, Y, overlay, cmap='gray', alpha=0.35)
    
    ax.axis([cx.min(), cx.max(), cy.max(), cy.min()])
    
    # Add colorbar
    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes("right", size="3%", pad=0.06)
    cbar = fig.colorbar(col1, cax=cax1, extend='max')
    cbar.ax.tick_params(labelsize=18)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='black')
    plt.close()

def draw_contour(
    image: Image.Image,
    mask: Image.Image,
    threshold: int = 128,
    color: Tuple[int, int, int] = (255, 255, 0)
) -> Image.Image:
    """
    Draw contour of the mask on the image.
    
    Args:
        image: Base image
        mask: Binary mask
        threshold: Threshold for mask binarization
        color: RGB color for the contour
        
    Returns:
        Image with drawn contour
    """
    # Convert mask to binary
    binary_mask = mask.point(lambda p: p >= threshold and 255)
    
    # Find edges
    edges = binary_mask.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges)
    
    # Convert image to RGBA
    result = np.array(image)
    
    # Add alpha channel if not present
    if len(result.shape) == 3 and result.shape[2] == 3:
        result = np.concatenate([result, np.full((*result.shape[:2], 1), 255)], axis=2)
    
    # Draw contour
    result[np.nonzero(edges_array)] = list(color) + [255]
    
    return Image.fromarray(result)

def overlay_maps(
    base_image: Image.Image,
    current_mask: Image.Image,
    past_mask: Image.Image,
    transparency: float = 0.4
) -> Image.Image:
    """
    Create overlay visualization of current and past masks.
    
    Args:
        base_image: Base image to overlay on
        current_mask: Current time point mask
        past_mask: Previous time point mask
        transparency: Transparency level for overlay
        
    Returns:
        Image with overlay visualization
    """
    base_image = base_image.convert('RGBA')
    
    # Create combined state map
    current_array = np.array(current_mask)
    past_array = np.array(past_mask)
    state_map = (current_array > 0).astype(np.uint8) + 2 * (past_array > 0).astype(np.uint8)
    
    # Create overlay
    overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    overlay_array = np.array(overlay)
    
    # Apply colors
    for state, color in COLOR_MAP.items():
        alpha = 0 if (state == 0 or state == 3) else transparency
        overlay_array[state_map == state] = list(color) + [int(alpha * 255)]
    
    overlay = Image.fromarray(overlay_array)
    
    return Image.alpha_composite(base_image, overlay)

def save_visualization_results(
    session_path: str,
    file_prefix: str,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray]
) -> None:
    """
    Save all visualization results for a session.
    
    Args:
        session_path: Path to save session results
        file_prefix: Prefix for saved files
        images: Dictionary containing different image arrays 3, h, w
        masks: Dictionary containing different mask arrays  4, h, w
    """
    create_directory(session_path)
    
    # Save ground truth and predicted masks
    for mask_name, masks_4session in masks.items():
        for j in range(4):
            mask_data = masks_4session[j, :, :].astype(float)
            mask_image = to_pil(mask_data)
            mask_image.save(os.path.join(session_path, f"{file_prefix}-mask-sess{j}-{mask_name}.png"))
        
    # Save images with overlays and contours
    for img_name, img_3modal in images.items():
        print(f'image: {img_name}, img_3modal.shape: {img_3modal.shape}')
        for i in range(3):  # For each modality
            img_data = img_3modal[i, :, :].astype(float)
            base_image = to_pil(img_data)
            
            # Save original image
            base_image.save(os.path.join(session_path, f"{file_prefix}-image-modal{i}-{img_name}.png"))
        
        # Create and save overlay
        if 'gt_mask' in masks and 'ref_mask' in masks:
            overlay_image = overlay_maps(base_image, masks['gt_mask'], masks['ref_mask'])
            overlay_image.save(os.path.join(session_path, f"{file_prefix}-{img_name}_overlay.png"))
        
        # Create and save contour
        if 'gt_mask' in masks:
            contour_image = draw_contour(base_image, Image.fromarray(masks['gt_mask']))
            contour_image.save(os.path.join(session_path, f"{file_prefix}-{img_name}_contour.png"))