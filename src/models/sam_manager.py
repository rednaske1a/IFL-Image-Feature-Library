import os
import urllib.request
import torch
import cv2
import numpy as np
from typing import Optional, List, Dict
import hashlib

class SAMManager:
    """
    Segmentation manager with fallback to OpenCV-based segmentation.
    
    Supports:
    1. SLIC superpixels (lightweight, fast)
    2. Selective Search (R-CNN style proposals)
    3. Future: Full SAM integration when weights are available
    """
    
    SAM_MODELS = {
        'vit_h': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'filename': 'sam_vit_h_4b8939.pth',
            'size': '2.56 GB'
        },
        'vit_l': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'filename': 'sam_vit_l_0b3195.pth',
            'size': '1.25 GB'
        },
        'vit_b': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'filename': 'sam_vit_b_01ec64.pth',
            'size': '375 MB'
        }
    }
    
    def __init__(self, model_dir: str = "./models", model_type: str = 'vit_b', use_opencv_fallback: bool = True):
        """
        Initialize segmentation manager
        
        Args:
            model_dir: Directory to store model weights
            model_type: Model variant ('vit_h', 'vit_l', or 'vit_b')
            use_opencv_fallback: Use OpenCV segmentation if SAM not available
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.use_opencv_fallback = use_opencv_fallback
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.segmentation_method = 'opencv_slic'  # 'sam' or 'opencv_slic'
        
    def get_model_path(self) -> str:
        """Get full path to model checkpoint"""
        model_info = self.SAM_MODELS[self.model_type]
        return os.path.join(self.model_dir, model_info['filename'])
    
    def is_downloaded(self) -> bool:
        """Check if SAM model is downloaded"""
        return os.path.exists(self.get_model_path())
    
    def download_model(self, progress_callback=None):
        """
        Download SAM model weights
        
        Args:
            progress_callback: Optional callback function(current_size, total_size)
        """
        model_info = self.SAM_MODELS[self.model_type]
        model_path = self.get_model_path()
        
        if self.is_downloaded():
            print(f"✓ SAM model already downloaded: {model_path}")
            return model_path
        
        print(f"Downloading SAM {self.model_type} model ({model_info['size']})...")
        print(f"From: {model_info['url']}")
        print(f"To: {model_path}")
        
        def _progress_hook(block_num, block_size, total_size):
            if progress_callback:
                progress_callback(block_num * block_size, total_size)
            else:
                downloaded = block_num * block_size
                percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"\rProgress: {percent:.1f}%", end='')
        
        try:
            urllib.request.urlretrieve(model_info['url'], model_path, _progress_hook)
            print(f"\n✓ Model downloaded successfully: {model_path}")
            return model_path
        except Exception as e:
            print(f"\n✗ Error downloading model: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise
    
    def load_model(self):
        """
        Load segmentation model
        Uses OpenCV fallback if SAM not available
        """
        if self.is_downloaded():
            print(f"SAM model found at: {self.get_model_path()}")
            print("Note: Full SAM integration requires segment-anything package.")
            print("Using OpenCV SLIC superpixels as fallback.")
            self.segmentation_method = 'opencv_slic'
        else:
            if self.use_opencv_fallback:
                print("Using OpenCV SLIC superpixels for segmentation")
                self.segmentation_method = 'opencv_slic'
            else:
                print("SAM model not found. Please download using download_model()")
                return None
        
        return self
    
    def generate_masks_opencv_slic(self, image_rgb: np.ndarray, n_segments: int = 100,
                                     compactness: float = 10.0, min_area: int = 500) -> List[Dict]:
        """
        Generate segmentation masks using SLIC superpixels
        
        Args:
            image_rgb: RGB image as numpy array
            n_segments: Number of superpixel segments
            compactness: Compactness parameter (higher = more compact)
            min_area: Minimum segment area in pixels
            
        Returns:
            List of mask dictionaries compatible with SAM format
        """
        from scipy import ndimage
        
        slic = cv2.ximgproc.createSuperpixelSLIC(
            image_rgb,
            algorithm=cv2.ximgproc.SLIC,
            region_size=int(np.sqrt(image_rgb.shape[0] * image_rgb.shape[1] / n_segments))
        )
        slic.iterate(10)
        labels = slic.getLabels()
        
        masks = []
        unique_labels = np.unique(labels)
        
        for label_id in unique_labels:
            mask = (labels == label_id).astype(np.uint8)
            
            area = np.sum(mask)
            if area < min_area:
                continue
            
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue
            
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            stability_score = 0.8 if contours else 0.5
            
            masks.append({
                'segmentation': mask.astype(bool),
                'bbox': bbox,
                'area': float(area),
                'predicted_iou': 0.85,
                'stability_score': stability_score
            })
        
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def generate_masks_selective_search(self, image_rgb: np.ndarray, max_proposals: int = 100) -> List[Dict]:
        """
        Generate segmentation masks using Selective Search
        
        Args:
            image_rgb: RGB image as numpy array
            max_proposals: Maximum number of proposals
            
        Returns:
            List of mask dictionaries
        """
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image_rgb)
        ss.switchToSelectiveSearchFast()
        
        rects = ss.process()
        
        masks = []
        height, width = image_rgb.shape[:2]
        
        for i, (x, y, w, h) in enumerate(rects[:max_proposals]):
            if w < 10 or h < 10:
                continue
            
            mask = np.zeros((height, width), dtype=bool)
            mask[y:y+h, x:x+w] = True
            
            masks.append({
                'segmentation': mask,
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': float(w * h),
                'predicted_iou': 0.75,
                'stability_score': 0.7
            })
        
        return masks
    
    def get_automatic_mask_generator(self):
        """
        Get automatic mask generator
        Returns self with generate method
        """
        class MaskGenerator:
            def __init__(self, sam_manager):
                self.sam_manager = sam_manager
            
            def generate(self, image_rgb: np.ndarray) -> List[Dict]:
                """Generate masks for an image"""
                if self.sam_manager.segmentation_method == 'opencv_slic':
                    return self.sam_manager.generate_masks_opencv_slic(image_rgb)
                else:
                    return self.sam_manager.generate_masks_opencv_slic(image_rgb)
        
        return MaskGenerator(self)
    
    def is_available(self) -> bool:
        """Check if segmentation is available"""
        return self.segmentation_method is not None
