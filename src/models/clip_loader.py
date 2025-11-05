"""
Simple CLIP loader that downloads model weights directly
"""
import torch
import torch.nn as nn
import os
import urllib.request
import warnings
from PIL import Image
import numpy as np

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    
    if os.path.exists(download_target):
        return download_target
    
    print(f"Downloading {url} to {download_target}")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        output.write(source.read())
    
    return download_target

def load(name: str = "ViT-B/32", device: str = "cpu", download_root: str = "~/.cache/clip"):
    """
    Load CLIP model
    
    Args:
        name: model name
        device: device to load model on
        download_root: directory to download model weights to
        
    Returns:
        tuple: (model, preprocess_fn)
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], os.path.expanduser(download_root))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")
    
    try:
        with open(model_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location=device).eval()
            state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    if hasattr(model, 'visual'):
        model = model.to(device)
    
    # Create preprocessing function
    def preprocess(image):
        """Preprocess image for CLIP"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Resize and center crop
        image = image.convert("RGB")
        image = image.resize((224, 224), Image.BICUBIC)
        
        # Convert to tensor and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image_np = (image_np - mean) / std
        
        # Convert to torch tensor (C, H, W)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        
        return image_tensor
    
    return model, preprocess

def tokenize(texts, context_length: int = 77):
    """
    Tokenize text for CLIP
    
    Args:
        texts: str or list of str
        context_length: maximum text length
        
    Returns:
        torch.LongTensor of shape (batch_size, context_length)
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Simple tokenization (this is a placeholder - real CLIP uses BPE)
    # For a production system, you'd need the full tokenizer
    # For now, we'll create a simple version
    
    import re
    
    tokens = []
    for text in texts:
        # Simple word-based tokenization
        words = re.findall(r'\w+', text.lower())
        # Truncate to context length (leaving room for special tokens)
        words = words[:context_length-2]
        # Convert to token IDs (placeholder - real implementation uses vocab)
        token_ids = [0] + [hash(w) % 49407 for w in words] + [0]
        # Pad to context length
        token_ids = token_ids + [0] * (context_length - len(token_ids))
        tokens.append(token_ids[:context_length])
    
    return torch.LongTensor(tokens)
