import cv2
import numpy as np
from PIL import Image
import os
import torch
import urllib.request
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class PhotoEnhancer:
    def __init__(self):
        """Initialize the AI model with auto GPU/CPU detection"""
        
        # Auto-detect GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_half = True if torch.cuda.is_available() else False
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                       num_block=23, num_grow_ch=32, scale=4)
        
        # Download model if not exists
        model_path = 'models/RealESRGAN_x4plus.pth'
        os.makedirs('models', exist_ok=True)
        
        if not os.path.exists(model_path):
            print("Downloading AI model (65MB)... Please wait...")
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=400 if self.device.type == 'cpu' else 0,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=self.device
        )
    
    def enhance(self, input_path, output_path):
        """Enhance image quality using AI"""
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("Could not read image")
        
        # AI Enhancement
        output, _ = self.upsampler.enhance(img, outscale=4)
        
        # Save result
        cv2.imwrite(output_path, output)
        return output_path
    
    def denoise(self, img):
        """Remove noise from image"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    def auto_adjust(self, img):
        """Auto adjust brightness and contrast"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l,a,b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)