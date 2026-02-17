import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import urllib.request

class SmartEnhancer:
    def __init__(self):
        """Initialize AI models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_half = True if torch.cuda.is_available() else False
        
        print(f"Using device: {self.device}")
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                       num_block=23, num_grow_ch=32, scale=4)
        
        model_path = 'models/RealESRGAN_x4plus.pth'
        os.makedirs('models', exist_ok=True)
        
        if not os.path.exists(model_path):
            print("Downloading AI model (65MB)...")
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded!")
        
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
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def ai_detect_optimal_scale(self, img_path):
        """AI determines best upscale factor based on image analysis"""
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        total_pixels = h * w
        
        # Analyze image quality
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Detect compression artifacts
        jpeg_quality = self.estimate_jpeg_quality(img)
        
        # Noise level
        noise_level = np.std(gray)
        
        # AI Decision Logic
        scale = 4  # Default
        reason = []
        
        # Rule 1: Resolution check
        if total_pixels > 3000000:  # > 3MP
            scale = 2
            reason.append(f"High resolution ({w}x{h})")
        elif total_pixels > 1500000:  # > 1.5MP
            scale = 3
            reason.append(f"Medium-high resolution ({w}x{h})")
        
        # Rule 2: Quality check
        if jpeg_quality < 60:
            scale = min(scale, 2)
            reason.append(f"Low JPEG quality ({jpeg_quality}%)")
        
        # Rule 3: Artifact check
        if blur_score < 200 and noise_level > 40:
            scale = min(scale, 3)
            reason.append("Detected artifacts/noise")
        
        # Rule 4: Very blurry images
        if blur_score < 100:
            scale = min(scale, 2)
            reason.append(f"Severe blur detected")
        
        print(f"AI Decision: {scale}x upscale")
        if reason:
            print(f"Reasoning: {', '.join(reason)}")
        
        return scale, reason
    
    def estimate_jpeg_quality(self, img):
        """Estimate JPEG compression quality"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check for blocking artifacts (8x8 DCT blocks)
        h, w = gray.shape
        block_size = 8
        
        block_diff = 0
        count = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                block_diff += np.std(block)
                count += 1
        
        avg_diff = block_diff / count if count > 0 else 50
        
        # Convert to quality estimate (0-100)
        quality = min(100, max(0, int(100 - (avg_diff / 2))))
        
        return quality
    
    def ai_denoise(self, img):
        """AI-powered denoising"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    def ai_sharpen(self, img):
        """AI sharpening for blurry images"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)
    
    def ai_brighten(self, img):
        """Intelligent brightness adjustment"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l,a,b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def enhance_faces(self, img, faces):
        """AI-powered face enhancement"""
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.detailEnhance(face, sigma_s=10, sigma_r=0.15)
            img[y:y+h, x:x+w] = face
        
        return img
    
    def auto_enhance(self, input_path, output_path, ai_analysis):
        """AI-Powered Auto Enhancement with intelligent scaling"""
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("Could not read image")
        
        # AI determines optimal scale
        optimal_scale, scale_reasons = self.ai_detect_optimal_scale(input_path)
        
        # Step 1: Pre-processing
        if "AI-powered sharpening" in ai_analysis.get('recommendations', []):
            print("AI: Applying sharpening...")
            img = self.ai_sharpen(img)
        
        if "Intelligent brightness boost" in ai_analysis.get('recommendations', []):
            print("AI: Adjusting brightness...")
            img = self.ai_brighten(img)
        
        # Step 2: Face enhancement
        if ai_analysis.get('has_faces', False):
            print(f"AI: Enhancing {ai_analysis['face_count']} face(s)...")
            faces = self.face_cascade.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5
            )
            img = self.enhance_faces(img, faces)
        
        # Save pre-enhanced
        temp_path = input_path.replace('.', '_preprocessed.')
        cv2.imwrite(temp_path, img)
        
        # Step 3: AI Super-Resolution with optimal scale
        print(f"AI: Applying {optimal_scale}x super-resolution...")
        output, _ = self.upsampler.enhance(img, outscale=optimal_scale)
        
        # Step 4: Post-processing to reduce artifacts
        print("AI: Reducing artifacts...")
        output = cv2.fastNlMeansDenoisingColored(output, None, 3, 3, 7, 21)
        
        if "Contrast enhancement" in ai_analysis.get('recommendations', []):
            print("AI: Enhancing contrast...")
            output = cv2.convertScaleAbs(output, alpha=1.1, beta=5)
        
        # Save final
        cv2.imwrite(output_path, output)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return output_path, optimal_scale, scale_reasons