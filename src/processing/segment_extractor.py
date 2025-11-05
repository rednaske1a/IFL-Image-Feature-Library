import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

class SegmentExtractor:
    def __init__(self, blur_kernel=21):
        """
        Initialize segment extractor with blurred background approach
        
        Args:
            blur_kernel: Gaussian blur kernel size (default 21, recommended by research)
        """
        self.blur_kernel = blur_kernel
    
    def extract_segment_with_blurred_background(self, mask_dict, img_rgb):
        """
        Extract segment with blurred background (research-backed approach).
        
        Based on CLIPAway, Mask-ControlNet, and industry standards (Ultralytics YOLOv8).
        Blur de-emphasizes background while preserving context for CLIP.
        
        Args:
            mask_dict: SAM mask dictionary with 'segmentation', 'bbox', 'area'
            img_rgb: Original image (RGB)
            
        Returns:
            dict: {
                'image': numpy array of cropped segment with blurred background
                'pil_image': PIL Image for CLIP preprocessing
                'bbox': [x, y, w, h] bounding box
                'area': mask area in pixels
                'iou': predicted IOU quality score
                'stability': stability score
            }
        """
        seg_mask = mask_dict['segmentation']
        x, y, w, h = mask_dict['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        crop = img_rgb[y:y+h, x:x+w].copy()
        
        if crop.size == 0:
            return None
        
        mask_crop = seg_mask[y:y+h, x:x+w]
        
        blurred = cv2.GaussianBlur(crop, (self.blur_kernel, self.blur_kernel), 0)
        
        result = crop.copy()
        result[~mask_crop] = blurred[~mask_crop]
        
        pil_image = Image.fromarray(result.astype(np.uint8))
        
        return {
            'image': result,
            'pil_image': pil_image,
            'bbox': [x, y, w, h],
            'area': float(mask_dict['area']),
            'iou': float(mask_dict['predicted_iou']),
            'stability': float(mask_dict['stability_score'])
        }
    
    def process_image(self, image_path, mask_generator, clip_model, progress_callback=None):
        """
        Process an image: generate segments and embeddings
        
        Args:
            image_path: path to image file
            mask_generator: SAM mask generator
            clip_model: ModelManager instance with CLIP loaded
            progress_callback: optional callback(current, total)
            
        Returns:
            tuple: (segments, embeddings)
                segments: list of segment dicts
                embeddings: numpy array of shape (N, 512)
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        masks = mask_generator.generate(img_rgb)
        
        segments = []
        embeddings = []
        
        for idx, mask in enumerate(masks):
            if progress_callback:
                progress_callback(idx, len(masks))
            
            seg_data = self.extract_segment_with_blurred_background(mask, img_rgb)
            
            if seg_data is None:
                continue
            
            embedding = clip_model.encode(seg_data['pil_image'])
            
            seg_data['id'] = idx
            
            segments.append(seg_data)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        return segments, embeddings
