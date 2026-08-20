import cv2
import numpy as np

class LowLightEnhancer:
    def __init__(self, low_light_threshold=100, clip_limit=2.0):
        """
        Initializes the LowLightEnhancer.
        
        Args:
            low_light_threshold (int): Average brightness threshold below which enhancement is applied.
                                       Range 0-255. Default 100.
            clip_limit (float): Threshold for contrast limiting in CLAHE. Default 2.0.
        """
        self.low_light_threshold = low_light_threshold
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    def enhance(self, frame):
        """
        Checks if the frame is dark and applies enhancement if needed.
        
        Args:
            frame: Input BGR image.
            
        Returns:
            Enhanced BGR image (or original if lighting is sufficient).
        """
        try:
            # Convert to LAB color space to separate lightness from color
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Calculate average brightness
            avg_brightness = np.mean(l)
            
            # Check if enhancement is needed
            if avg_brightness < self.low_light_threshold:
                # Apply CLAHE to the L-channel
                l_enhanced = self.clahe.apply(l)
                
                # Merge channels back
                lab_enhanced = cv2.merge((l_enhanced, a, b))
                
                # Convert back to BGR
                enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                
                # Optional: Add a small text indicator for debugging/demo
                cv2.putText(enhanced_frame, "Night Vision ON", (10, frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                return enhanced_frame
            
            return frame
            
        except Exception as e:
            print(f"Error in LowLightEnhancer: {e}")
            return frame
