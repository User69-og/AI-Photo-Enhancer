import cv2
import numpy as np

class ImageAnalytic:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)

    def get_action_plan(self):
        if self.img is None: return "INVALID"
        
        # Check for blur/detail loss
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check for "fucked up" factors (high noise or empty regions)
        is_damaged = np.std(self.img) < 15 or score < 80
        
        if is_damaged:
            return "REGENERATE" # Needs Diffusion Hallucination
        return "ENHANCE"       # Needs Real-ESRGAN Upscaling