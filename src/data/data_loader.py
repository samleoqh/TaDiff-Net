"""
Data loading utilities for TaDiff model.
Handles dataset creation, transformations, and data loading operations.
"""
import os
from typing import List, Dict, Optional

import torch
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, MapTransform, 
    CenterSpatialCropd, SpatialPadd, CropForegroundd
)

class MergeMultiLabels(MapTransform):
    """Convert multi-class labels to binary labels.
    
    Args:
        keys: Keys of the corresponding items to be transformed
    """
    def __call__(self, data: Dict) -> Dict:
        """
        Args:
            data: Dictionary containing the data to transform
            
        Returns:
            Transformed data dictionary with binary labels
        """
        for key in self.key_iterator(data):
            data[key] = data[key] > 0  # merge all labels 1, 2, 3, ... to be 1
        return data

def get_transform_pipeline() -> Compose:
    """
    Create the data transformation pipeline.
    
    Returns:
        Composed transformation pipeline
    """
    npz_keys = ['image', 'label', 'days', 'treatment']
    
    return Compose([
        LoadImaged(keys=npz_keys, image_only=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[192, 192, 192]),
        SpatialPadd(keys=["image", "label"], spatial_size=(192, 192, 192)),
        MergeMultiLabels(keys=["label"]),
    ])


def load_data(test_file_list: Optional[List[Dict[str, str]]] = None) -> DataLoader:
    """
    Create data loader for test data.
    
    Args:
        test_file_list: List of dictionaries containing file paths.
                       If None, uses default test files.
                       
    Returns:
        DataLoader instance for the test dataset
    """
    if test_file_list is None:
        print('No test file list provided. Using default test files.')
        # test_file_list = get_test_files()
        return 0
        
    transform = get_transform_pipeline()
    test_dataset = CacheDataset(data=test_file_list, transform=transform)
    
    # Only support bs=1, num_worker=0 for support across platforms
    return DataLoader(test_dataset, batch_size=1, shuffle=False)
