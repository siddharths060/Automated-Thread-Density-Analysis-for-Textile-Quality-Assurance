"""
Thread Density Analysis Service

This module provides functionality to analyze thread density in textile images.
It uses image processing techniques to detect and count threads.
"""
import os
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans
from typing import Dict, Any, Tuple, Optional
import uuid

from utils.image_processing import preprocess_image, enhance_contrast, resize_image
from models.thread_detector import ThreadDetector
from utils.logging_config import get_logger

# Initialize logger
logger = get_logger("thread_analysis")


class ThreadAnalysisService:
    """Service for analyzing thread density in textile images."""
    
    def __init__(self, detector: Optional[ThreadDetector] = None, pixels_per_cm: float = 40.0):
        """
        Initialize the thread analysis service.
        
        Args:
            detector: Thread detector instance (optional, will create new one if not provided)
            pixels_per_cm: Calibration value for pixels per centimeter
        """
        self.thread_detector = detector or ThreadDetector()
        self.pixels_per_cm = pixels_per_cm
        logger.info("Thread analysis service initialized with calibration: %.2f px/cm", pixels_per_cm)
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze thread density in the provided image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing analysis results including thread counts and annotated image
            
        Raises:
            ValueError: If the image cannot be loaded or processed
            IOError: If there are file operation issues
        """
        start_time = time.time()
        logger.info(f"Starting analysis of image: {image_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Resize large images for processing efficiency
            orig_shape = image.shape
            if max(image.shape[0], image.shape[1]) > 1024:
                image, scale_factor = resize_image(image)
                logger.info(f"Image resized by factor {scale_factor:.2f} for processing efficiency")
            else:
                scale_factor = 1.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance image for better thread detection
            enhanced = enhance_contrast(gray)
            logger.debug("Image preprocessed and enhanced for thread detection")
            
            # Detect threads using the model
            warp_threads, weft_threads = self.detect_threads(enhanced)
            logger.info(f"Detected {len(warp_threads)} warp threads and {len(weft_threads)} weft threads")
            
            # Create annotated image
            annotated_image = self.annotate_threads(image, warp_threads, weft_threads)
            
            # Calculate thread density using calibration
            warp_density = len(warp_threads) / (image.shape[1] / self.pixels_per_cm)
            weft_density = len(weft_threads) / (image.shape[0] / self.pixels_per_cm)
            avg_density = (warp_density + weft_density) / 2
            
            # Scale thread counts if image was resized
            if scale_factor != 1.0:
                logger.debug(f"Adjusting thread counts for original image dimensions (scale factor: {scale_factor:.2f})")
            
            # Save annotated image
            result_dir = "results"
            os.makedirs(result_dir, exist_ok=True)
            result_file = os.path.join(result_dir, f"{uuid.uuid4()}.jpg")
            
            if not cv2.imwrite(result_file, annotated_image):
                logger.error(f"Failed to save result image to {result_file}")
                raise IOError(f"Failed to save result image to {result_file}")
                
            logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
            
            # Return results
            return {
                "thread_count": {
                    "warp_count": len(warp_threads),
                    "weft_count": len(weft_threads),
                    "total_count": len(warp_threads) + len(weft_threads),
                    "density": round(avg_density, 2),
                    "unit": "cm"
                },
                "annotated_image_path": result_file,
                "processing_time": round(time.time() - start_time, 2)
            }
            
        except FileNotFoundError as e:
            logger.error(f"File error during thread analysis: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Value error during thread analysis: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during thread analysis: {str(e)}", exc_info=True)
            raise ValueError(f"Thread analysis failed: {str(e)}")
    
    def detect_threads(self, image: np.ndarray) -> Tuple[list, list]:
        """
        Detect threads in the image using gradient analysis.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Tuple of (warp_threads, weft_threads) where each is a list of thread positions
            
        Note:
            Warp threads are vertical, weft threads are horizontal
        """
        logger.debug("Starting thread detection")
        height, width = image.shape
        
        try:
            # For horizontal threads (weft) - detect vertical edges
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            
            # For vertical threads (warp) - detect horizontal edges
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Find vertical thread positions (warp)
            h_projection = np.sum(grad_y, axis=0)
            
            # Determine optimal window size for smoothing based on image width
            smooth_window = min(15, width // 10)
            if smooth_window % 2 == 0:  # Must be odd for Savitzky-Golay filter
                smooth_window += 1
                
            # Apply smoothing filter
            h_projection = signal.savgol_filter(h_projection, smooth_window, 3)
            
            # Use peak finding with dynamic distance based on image dimensions
            min_distance = max(5, width // 100)
            peaks_warp, _ = signal.find_peaks(h_projection, distance=min_distance, prominence=0.1)
            
            # Find horizontal thread positions (weft)
            v_projection = np.sum(grad_x, axis=1)
            
            # Determine optimal window size for smoothing based on image height
            smooth_window = min(15, height // 10)
            if smooth_window % 2 == 0:  # Must be odd for Savitzky-Golay filter
                smooth_window += 1
                
            v_projection = signal.savgol_filter(v_projection, smooth_window, 3)
            
            # Dynamic distance for peak finding
            min_distance = max(5, height // 100)
            peaks_weft, _ = signal.find_peaks(v_projection, distance=min_distance, prominence=0.1)
            
            # Log detection results
            logger.debug(f"Detected {len(peaks_warp)} warp threads and {len(peaks_weft)} weft threads")
            
            return peaks_warp.tolist(), peaks_weft.tolist()
            
        except Exception as e:
            logger.error(f"Error in thread detection: {str(e)}", exc_info=True)
            raise ValueError(f"Thread detection failed: {str(e)}")
        
        return peaks_warp.tolist(), peaks_weft.tolist()
    
    def annotate_threads(self, image: np.ndarray, warp_threads: list, weft_threads: list) -> np.ndarray:
        """
        Create an annotated image with threads highlighted.
        
        Args:
            image: Original image
            warp_threads: List of warp thread positions
            weft_threads: List of weft thread positions
            
        Returns:
            Annotated image with threads highlighted
        """
        try:
            logger.debug("Creating annotated image visualization")
            annotated = image.copy()
            height, width = annotated.shape[:2]
            
            # Create semi-transparent overlay for better thread visibility
            overlay = annotated.copy()
            
            # Draw horizontal threads (weft) - in blue with slight transparency
            for y in weft_threads:
                if 0 <= y < height:  # Ensure within bounds
                    cv2.line(overlay, (0, y), (width, y), (255, 0, 0), 1)
            
            # Draw vertical threads (warp) - in green with slight transparency
            for x in warp_threads:
                if 0 <= x < width:  # Ensure within bounds
                    cv2.line(overlay, (x, 0), (x, height), (0, 255, 0), 1)
            
            # Apply overlay with transparency
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            
            # Calculate position for text based on image size
            margin = 10
            font_size = max(0.5, min(1.0, width / 640))  # Scale based on image width
            thickness = max(1, int(font_size * 2))
            line_height = int(30 * font_size)
            
            # Add information box for results
            info_box_height = 4 * line_height
            cv2.rectangle(annotated, (margin-5, margin-5), 
                        (300, margin + info_box_height), 
                        (0, 0, 0), cv2.FILLED)
            cv2.rectangle(annotated, (margin-5, margin-5), 
                        (300, margin + info_box_height), 
                        (255, 255, 255), 1)
            
            # Add labels for thread counts with improved formatting
            cv2.putText(annotated, f"Warp: {len(warp_threads)} threads (vertical)", 
                        (margin, margin + line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
            
            cv2.putText(annotated, f"Weft: {len(weft_threads)} threads (horizontal)", 
                        (margin, margin + 2*line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), thickness)
            
            cv2.putText(annotated, f"Total: {len(warp_threads) + len(weft_threads)} threads", 
                        (margin, margin + 3*line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), thickness)
                        
            # Add legend
            legend_margin = 10
            legend_y = height - 3*line_height - legend_margin
            legend_width = 200
            legend_height = 3*line_height
            
            cv2.rectangle(annotated, 
                        (width - legend_width - legend_margin, legend_y - legend_margin),
                        (width - legend_margin, legend_y + legend_height),
                        (0, 0, 0), cv2.FILLED)
                        
            cv2.rectangle(annotated, 
                        (width - legend_width - legend_margin, legend_y - legend_margin),
                        (width - legend_margin, legend_y + legend_height),
                        (255, 255, 255), 1)
                        
            # Add legend text
            cv2.putText(annotated, "Legend:", 
                      (width - legend_width - legend_margin + 10, legend_y + line_height - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                      
            # Green line for warp
            x_start = width - legend_width - legend_margin + 10
            x_end = x_start + 40
            y_line = legend_y + line_height
            cv2.line(annotated, (x_start, y_line), (x_end, y_line), (0, 255, 0), 2)
            cv2.putText(annotated, "Warp threads", 
                      (x_end + 10, y_line + 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                      
            # Blue line for weft
            y_line = legend_y + 2*line_height
            cv2.line(annotated, (x_start, y_line), (x_end, y_line), (255, 0, 0), 2)
            cv2.putText(annotated, "Weft threads", 
                      (x_end + 10, y_line + 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                      
            return annotated
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {str(e)}")
            # Return original image if annotation fails
            return image
