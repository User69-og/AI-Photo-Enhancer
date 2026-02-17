import cv2
import numpy as np
from PIL import Image
import pytesseract

class AIAnalyzer:
    def __init__(self):
        # Load pre-trained AI models
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, img):
        """AI Face Detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def detect_text(self, img_path):
        """AI Text Detection"""
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
            has_text = len(text.strip()) > 10
            return has_text, len(text.strip())
        except:
            return False, 0
    
    def classify_scene(self, img):
        """AI Scene Classification"""
        # Analyze image properties
        brightness = np.mean(img)
        contrast = np.std(img)
        
        # Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1])
        
        # Edge detection for detail level
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # AI Classification Logic
        if edge_density > 0.15:
            return "High Detail / Landscape"
        elif saturation > 100:
            return "Vibrant / Colorful"
        elif brightness < 100:
            return "Low Light / Night"
        elif contrast < 30:
            return "Flat / Low Contrast"
        else:
            return "General / Balanced"
    
    def predict_improvement(self, img_path):
        """ML-based Quality Improvement Prediction"""
        img = cv2.imread(img_path)
        
        # Feature extraction for prediction
        brightness = np.mean(img)
        contrast = np.std(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Simple ML prediction model (rule-based for now)
        improvement = 0
        
        if blur_score < 500:
            improvement += 30  # Will improve significantly
        if brightness < 100:
            improvement += 25
        if contrast < 40:
            improvement += 20
        
        # Add base improvement from upscaling
        improvement += 25
        
        return min(improvement, 95)  # Cap at 95%
    
    def recommend_enhancement_mode(self, img, faces):
        """AI Enhancement Mode Recommendation"""
        has_faces = len(faces) > 0
        brightness = np.mean(img)
        
        # Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1])
        
        # Text detection attempt
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text_pixels = np.count_nonzero(binary) / binary.size
        
        # AI Decision Tree
        if has_faces:
            return "Portrait Mode", 92
        elif text_pixels > 0.6 or text_pixels < 0.2:
            return "Document Mode", 88
        elif saturation > 100:
            return "Landscape Mode", 85
        else:
            return "General Mode", 80
    
    def analyze_content(self, img_path):
        """Complete AI-Powered Analysis"""
        img = cv2.imread(img_path)
        
        # AI Feature Detection
        faces = self.detect_faces(img)
        has_text, text_length = self.detect_text(img_path)
        scene_type = self.classify_scene(img)
        predicted_improvement = self.predict_improvement(img_path)
        enhancement_mode, mode_confidence = self.recommend_enhancement_mode(img, faces)
        
        # Generate AI Recommendations
        recommendations = []
        brightness = np.mean(img)
        contrast = np.std(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if len(faces) > 0:
            recommendations.append(f"Face enhancement ({len(faces)} face(s) detected)")
        if brightness < 100:
            recommendations.append("Intelligent brightness boost")
        if blur_score < 500:
            recommendations.append("AI-powered sharpening")
        if contrast < 40:
            recommendations.append("Contrast enhancement")
        if has_text:
            recommendations.append("Text clarity optimization")
        
        # Detect issues
        issues = []
        if blur_score < 300:
            issues.append("Severe blur detected")
        if brightness < 60:
            issues.append("Very dark image")
        if contrast < 25:
            issues.append("Low contrast")
        
        return {
            'scene_type': scene_type,
            'has_faces': len(faces) > 0,
            'face_count': len(faces),
            'has_text': has_text,
            'text_length': text_length,
            'enhancement_mode': enhancement_mode,
            'mode_confidence': mode_confidence,
            'predicted_improvement': predicted_improvement,
            'recommendations': recommendations,
            'issues': issues,
            'overall_confidence': mode_confidence
        }