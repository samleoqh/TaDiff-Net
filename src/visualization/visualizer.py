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

class Visualizer:
    def __init__(self, colors: dict, bg_color='black', fg_color='white'):
        self.colors = colors
        self.bg_color = bg_color
        self.fg_color = fg_color
        
        # Set matplotlib parameters
        mpl.rc('image', cmap='gray')
        plt.rcParams["text.color"] = fg_color
        plt.rcParams["axes.labelcolor"] = fg_color
        plt.rcParams["xtick.color"] = fg_color
        plt.rcParams["ytick.color"] = fg_color
    
    def to_pil(self, arr, to_rgb=False):
        """Convert numpy array to PIL Image"""
        normalized = ((arr - arr.min()) / (arr.max() - arr.min() + 1.e-8)) * 255.9
        if to_rgb:
            return Image.fromarray(normalized.astype(np.uint8)).convert('RGB')
        return Image.fromarray(normalized.astype(np.uint8))
    
    def plot_uncertainty(self, arr, save_path, overlay=None):
        """Plot uncertainty map with optional overlay"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        cx = np.arange(arr.shape[1])
        cy = np.arange(arr.shape[0])
        X, Y = np.meshgrid(cx, cy)
        
        ax.set_aspect('equal')
        col1 = ax.pcolormesh(X, Y, arr, cmap='magma', 
                            vmin=arr.min(), vmax=arr.max())
        
        if overlay is not None:
            ax.pcolormesh(X, Y, overlay, cmap='gray', alpha=0.35)
        
        ax.axis([cx.min(), cx.max(), cy.max(), cy.min()])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.06)
        cbar = fig.colorbar(col1, cax=cax, extend='max')
        cbar.ax.tick_params(labelsize=18)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor=self.bg_color)
        plt.close()
    
    def draw_contour(self, img, mask, rgb=(255, 0, 0)):
        """Draw contour of mask on image"""
        # Create binary mask
        binary_mask = mask.point(lambda p: p >= 128 and 255)
        
        # Find edges
        edges = binary_mask.filter(ImageFilter.FIND_EDGES)
        edges_np = np.array(edges)
        
        # Convert image to RGBA if not already
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert('RGBA')
        img_np = np.array(img)
        
        # Draw edges in specified color
        img_np[np.nonzero(edges_np)] = [*rgb, 255]
        
        return Image.fromarray(img_np)
    
    def overlay_maps(self, img, curr_mask, past_mask, transparency=0.4):
        """Overlay current and past masks on image"""
        img = img.convert('RGBA')
        combined_mask = self.to_binary(np.array(curr_mask)) + 2 * self.to_binary(np.array(past_mask))
        
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay = np.array(overlay)
        
        for k, v in self.colors.items():
            alpha = 0 if (k == 0 or k == 3) else transparency
            overlay[combined_mask == k] = [*v, int(alpha * 255)]
        
        overlay = Image.fromarray(np.uint8(overlay))
        return Image.alpha_composite(img, overlay)
    
    @staticmethod
    def to_binary(arr, threshold=0.5):
        """Convert array to binary mask"""
        th = arr.max() * threshold
        return (arr >= th).astype(np.uint8)

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

