import cv2
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class ColorRange:
    name: str
    lower: np.ndarray
    upper: np.ndarray

class ColorSegmenter:
    COLORS = {
        'red': [ColorRange('red_low', np.array([0, 100, 100]), np.array([10, 255, 255])),
                ColorRange('red_high', np.array([170, 100, 100]), np.array([180, 255, 255]))],
        'green': [ColorRange('green', np.array([35, 100, 100]), np.array([85, 255, 255]))],
        'blue': [ColorRange('blue', np.array([100, 100, 100]), np.array([130, 255, 255]))],
        'yellow': [ColorRange('yellow', np.array([20, 100, 100]), np.array([30, 255, 255]))],
    }
    
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.masks: Dict[str, np.ndarray] = {}
    
    def detect(self, color_name: str) -> Tuple[np.ndarray, float]:
        if color_name not in self.COLORS:
            raise ValueError(f"Unknown color: {color_name}")
        mask = np.zeros(self.hsv.shape[:2], dtype=np.uint8)
        for cr in self.COLORS[color_name]:
            mask_part = cv2.inRange(self.hsv, cr.lower, cr.upper)
            mask = cv2.bitwise_or(mask, mask_part)
        percentage = (np.count_nonzero(mask) / mask.size) * 100
        self.masks[color_name] = mask
        return mask, percentage
    
    def custom_color(self, lower: Tuple[int, int, int], upper: Tuple[int, int, int], name: str = 'custom') -> np.ndarray:
        mask = cv2.inRange(self.hsv, np.array(lower), np.array(upper))
        self.masks[name] = mask
        return mask
    
    def morphology(self, mask_name: str, operation: str = 'close', kernel_size=(5, 5)) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        ops = {'close': cv2.MORPH_CLOSE, 'open': cv2.MORPH_OPEN}
        result = cv2.morphologyEx(self.masks[mask_name], ops.get(operation, cv2.MORPH_CLOSE), kernel)
        self.masks[mask_name] = result
        return result
    
    def extract(self, mask_name: str) -> np.ndarray:
        return cv2.bitwise_and(self.image, self.image, mask=self.masks[mask_name])
    
    def detect_all(self) -> Dict[str, float]:
        return {color: self.detect(color)[1] for color in self.COLORS.keys()}
    
    def stats(self, mask_name: str) -> Dict:
        contours, _ = cv2.findContours(self.masks[mask_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {}
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return {'area': cv2.contourArea(largest), 'width': w, 'height': h, 'position': (x, y)}