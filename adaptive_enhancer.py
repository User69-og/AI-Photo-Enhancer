import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import urllib.request

class AdaptiveEnhancer:
    def __init__(self):
        """Initialize multiple enhancement algorithms"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_half = True if torch.cuda.is_available() else False
        
        print(f"Using device: {self.device}")
        
        # Real-ESRGAN for natural photos
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
    
    def ai_classify_image_type(self, img_path):
        """AI classifies image type to choose best algorithm"""
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Feature extraction
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        brightness = np.mean(hsv[:,:,2])
        saturation = np.mean(hsv[:,:,1])
        saturation_std = np.std(hsv[:,:,1])
        
        noise_level = np.std(gray)
        
        # Check for compression artifacts
        dct_artifacts = self.detect_compression_artifacts(gray)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        has_faces = len(faces) > 0
        
        # AI Classification Logic
        image_type = "unknown"
        confidence = 0
        
        # Type 1: Natural Portrait
        if has_faces and blur_score > 300 and saturation_std < 50:
            image_type = "natural_portrait"
            confidence = 90
        
        # Type 2: Natural Landscape
        elif blur_score > 500 and edge_density < 0.15 and not has_faces:
            image_type = "natural_landscape"
            confidence = 85
        
        # Type 3: Artistic/Filtered Image
        elif saturation_std > 60 or edge_density > 0.2:
            image_type = "artistic_filtered"
            confidence = 80
        
        # Type 4: Low Quality/Compressed
        elif blur_score < 200 or dct_artifacts > 0.3:
            image_type = "low_quality_compressed"
            confidence = 85
        
        # Type 5: Screenshot/Document
        elif edge_density > 0.15 and saturation < 50:
            image_type = "screenshot_document"
            confidence = 75
        
        # Type 6: Dark/Low Light
        elif brightness < 80:
            image_type = "low_light"
            confidence = 80
        
        else:
            image_type = "general"
            confidence = 70
        
        print(f"AI Classification: {image_type} (confidence: {confidence}%)")
        
        return {
            'type': image_type,
            'confidence': confidence,
            'has_faces': has_faces,
            'face_count': len(faces),
            'blur_score': blur_score,
            'edge_density': edge_density,
            'brightness': brightness,
            'saturation': saturation
        }
    
    def detect_compression_artifacts(self, gray):
        """Detect JPEG compression artifacts"""
        h, w = gray.shape
        block_size = 8
        
        artifacts = 0
        total_blocks = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                block_var = np.var(block)
                
                # Low variance blocks indicate compression
                if block_var < 100:
                    artifacts += 1
                total_blocks += 1
        
        return artifacts / total_blocks if total_blocks > 0 else 0
    
    def enhance_bicubic(self, img, scale=2):
        """Traditional bicubic upscaling (safe for all images)"""
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    def enhance_lanczos(self, img, scale=2):
        """Lanczos upscaling (good for sharp images)"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w = img.shape[:2]
        pil_img = pil_img.resize((w * scale, h * scale), Image.LANCZOS)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def enhance_unsharp_mask(self, img):
        """Unsharp mask for sharpening without AI"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def enhance_detail_preserve(self, img):
        """Detail-preserving enhancement for artistic images"""
        # Use edge-preserving filter
        enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        
        # Gentle sharpening
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def enhance_brightness_adaptive(self, img):
        """Adaptive brightness enhancement"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l,a,b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def auto_enhance(self, input_path, output_path, ai_analysis):
        """AI selects and applies best enhancement algorithm"""
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("Could not read image")
        
        # AI classifies image type
        img_class = self.ai_classify_image_type(input_path)
        img_type = img_class['type']
        
        algorithm_used = ""
        processing_steps = []
        
        # AI chooses algorithm based on type
        if img_type == "natural_portrait":
            print("AI: Using Real-ESRGAN for natural portrait")
            algorithm_used = "Real-ESRGAN (Deep Learning)"
            processing_steps.append("Face-optimized super-resolution")
            
            # Light pre-processing
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
            processing_steps.append("Gentle denoising")
            
            # Real-ESRGAN 2x (safer)
            output, _ = self.upsampler.enhance(img, outscale=2)
            processing_steps.append("2x AI upscaling")
        
        elif img_type == "natural_landscape":
            print("AI: Using Real-ESRGAN for natural landscape")
            algorithm_used = "Real-ESRGAN (Deep Learning)"
            processing_steps.append("Landscape-optimized processing")
            
            # Real-ESRGAN 3x
            output, _ = self.upsampler.enhance(img, outscale=3)
            processing_steps.append("3x AI upscaling")
        
        elif img_type == "artistic_filtered":
            print("AI: Using Detail-Preserving algorithm (no AI upscaling)")
            algorithm_used = "Detail-Preserving Enhancement"
            processing_steps.append("Detected artistic/filtered content")
            
            # NO upscaling, just enhancement
            output = self.enhance_detail_preserve(img)
            processing_steps.append("Edge-aware enhancement")
            processing_steps.append("Gentle sharpening")
            
            # Optional 2x with traditional method
            if img.shape[0] * img.shape[1] < 500000:
                output = self.enhance_lanczos(output, scale=2)
                processing_steps.append("2x Lanczos upscaling (safe)")
        
        elif img_type == "low_quality_compressed":
            print("AI: Using Bicubic + Denoise (no AI upscaling)")
            algorithm_used = "Traditional Bicubic + Denoising"
            processing_steps.append("Detected compression artifacts")
            
            # Heavy denoise first
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            processing_steps.append("Heavy denoising")
            
            # Traditional upscaling
            output = self.enhance_bicubic(img, scale=2)
            processing_steps.append("2x Bicubic upscaling")
            
            # Sharpen slightly
            output = self.enhance_unsharp_mask(output)
            processing_steps.append("Unsharp mask")
        
        elif img_type == "screenshot_document":
            print("AI: Using Lanczos (text-optimized)")
            algorithm_used = "Lanczos (Text-Optimized)"
            processing_steps.append("Text/document mode")
            
            # Sharpen first
            output = self.enhance_unsharp_mask(img)
            processing_steps.append("Pre-sharpening")
            
            # Lanczos 2x
            output = self.enhance_lanczos(output, scale=2)
            processing_steps.append("2x Lanczos upscaling")
        
        elif img_type == "low_light":
            print("AI: Using Brightness Enhancement + Real-ESRGAN")
            algorithm_used = "Adaptive Brightness + AI"
            processing_steps.append("Low light detected")
            
            # Brighten first
            img = self.enhance_brightness_adaptive(img)
            processing_steps.append("Adaptive brightness boost")
            
            # Real-ESRGAN 2x
            output, _ = self.upsampler.enhance(img, outscale=2)
            processing_steps.append("2x AI upscaling")
        
        else:  # general
            print("AI: Using balanced approach")
            algorithm_used = "Balanced Enhancement"
            processing_steps.append("General image processing")
            
            # Light denoise
            img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
            processing_steps.append("Light denoising")
            
            # Real-ESRGAN 2x
            output, _ = self.upsampler.enhance(img, outscale=2)
            processing_steps.append("2x AI upscaling")
        
        # Save result
        cv2.imwrite(output_path, output)
        
        return output_path, algorithm_used, img_class, processing_steps