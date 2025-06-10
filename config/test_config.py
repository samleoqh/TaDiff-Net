from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TestConfig:
    # Data paths
    data_root: str = "./data/sailor"
    save_path: str = './sailor_eval_17'
    model_checkpoint: str = "./ckpt/tadiff_811.ckpt"


    # Patient IDs to test
    patient_ids: List[str] = field(default_factory=lambda: ['sub-17'])
    
    # Model parameters
    model_channels: int = 32
    num_heads: int = 1
    num_res_blocks: int = 1
    
    # Test parameters
    diffusion_steps: int = 600
    target_session_idx: int = 3
    num_samples: int = 5
    min_tumor_size: int = 20
    top_k_slices: int = 3
    
    
    # Data keys
    npz_keys: List[str] = field(default_factory=lambda: ['image', 'label', 'days', 'treatment'])
    
    # Visualization settings
    colors: dict = field(default_factory=lambda: {
        0: (0, 0, 0),      # background, black
        1: (0, 255, 0),    # class 1, green/ growth 
        2: (0, 0, 255),    # class 2, blue, shrinkage 
        3: (255, 0, 0),    # class 3, red / stable tumor
    })
    
    # Thresholds
    mask_threshold: float = 0.49
    dice_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75]) 