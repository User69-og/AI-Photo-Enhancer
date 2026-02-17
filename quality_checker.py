import cv2
import numpy as np

class QualityChecker:
    
    @staticmethod
    def assess_blur(img_path):
        """Detect blur level in image"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Adjusted thresholds for enhanced images
        if blur_score < 500:
            return "Blurry", blur_score
        elif blur_score < 1500:
            return "Moderate", blur_score
        else:
            return "Sharp", blur_score
    
    @staticmethod
    def assess_resolution(img_path):
        """Check image resolution"""
        img = cv2.imread(img_path)
        pixels = img.shape[0] * img.shape[1]
        
        if pixels < 500000:
            return "Low", pixels
        elif pixels < 2000000:
            return "Medium", pixels
        else:
            return "High", pixels
    
    @staticmethod
    def assess_brightness(img_path):
        """Check image brightness"""
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:,:,2])
        
        if brightness < 85:
            return "Dark", brightness
        elif brightness > 170:
            return "Bright", brightness
        else:
            return "Good", brightness
    
    @staticmethod
    def assess_noise(img_path):
        """Detect noise level"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        noise = np.std(img)
        
        if noise > 50:
            return "High Noise", noise
        elif noise > 30:
            return "Medium Noise", noise
        else:
            return "Low Noise", noise
    
    @staticmethod
    def overall_score(img_path):
        """Calculate overall quality score"""
        blur_status, blur_val = QualityChecker.assess_blur(img_path)
        res_status, res_val = QualityChecker.assess_resolution(img_path)
        bright_status, bright_val = QualityChecker.assess_brightness(img_path)
        
        # Better scoring that rewards high resolution
        blur_score = min(blur_val / 15, 100)  # Higher blur variance = sharper
        res_score = min(res_val / 20000, 100)  # Reward higher resolution
        bright_score = 100 - abs(bright_val - 127.5) / 1.275
        
        overall = (blur_score * 0.4 + res_score * 0.4 + bright_score * 0.2)
        
        return {
            'overall': round(overall, 2),
            'blur': blur_status,
            'resolution': res_status,
            'brightness': bright_status
        }