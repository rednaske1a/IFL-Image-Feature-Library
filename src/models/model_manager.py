import torch
from PIL import Image
import numpy as np
from src.vendored_clip import clip

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_preprocess = None
        self.sam_model = None
        self.sam_generator = None
        
    def load_clip(self, model_name="ViT-B/32"):
        """Load CLIP model for generating embeddings"""
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
            print(f"✓ CLIP model loaded on {self.device}")
        return self.clip_model, self.clip_preprocess
    
    def get_clip_model(self):
        """Get loaded CLIP model and preprocess function"""
        if self.clip_model is None:
            self.load_clip()
        return self.clip_model, self.clip_preprocess
    
    def is_clip_loaded(self):
        """Check if CLIP is loaded"""
        return self.clip_model is not None
    
    def encode(self, image_pil):
        """
        Encode a PIL image to CLIP embedding
        
        Args:
            image_pil: PIL Image
            
        Returns:
            numpy array: 512-dimensional normalized embedding
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded. Call load_clip() first")
        
        with torch.no_grad():
            image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
            embedding = self.clip_model.encode_image(image_input)
            embedding = embedding.cpu().numpy()[0]
            
        return embedding
    
    def encode_text(self, text):
        """
        Encode text to CLIP embedding
        
        Args:
            text: str or list of str
            
        Returns:
            numpy array: embedding(s)
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded")
        
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            text_tokens = clip.tokenize(text).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            embeddings = text_features.cpu().numpy()
            
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    def load_sam(self, checkpoint_path):
        """Load SAM model (optional - for full segmentation mode)"""
        try:
            from segment_anything import build_sam, SamAutomaticMaskGenerator
            self.sam_model = build_sam(checkpoint=checkpoint_path)
            self.sam_model.to(device=self.device)
            self.sam_generator = SamAutomaticMaskGenerator(self.sam_model)
            print(f"✓ SAM model loaded from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Could not load SAM: {e}")
            return False
    
    def get_sam_generator(self):
        """Get SAM mask generator"""
        return self.sam_generator
    
    def is_sam_loaded(self):
        """Check if SAM is loaded"""
        return self.sam_generator is not None
