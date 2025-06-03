"""
Thread detector model for identifying threads in textile images.

This module implements both a traditional computer vision approach and a U-Net
deep learning model for thread detection in textile images. The implementation 
automatically falls back to the traditional approach if the U-Net model isn't available.
"""
import os
import numpy as np
import cv2
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
from scipy import signal
import logging
import json

# Conditionally import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, BatchNorm2d, ReLU
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from utils.logging_config import get_logger

# Initialize logger
logger = get_logger("models.thread_detector")

# Default configuration for thread detection parameters
DEFAULT_CONFIG = {
    "detection": {
        "threshold": 0.5,
        "min_thread_length": 20,
        "min_thread_width": 1,
        "max_thread_width": 5,
        "peak_distance": 10,
        "peak_prominence": 0.1
    },
    "image_processing": {
        "resize_max_dimension": 1024,
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": (8, 8),
        "gaussian_blur_kernel": (5, 5),
        "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet means
        "normalize_std": [0.229, 0.224, 0.225]    # ImageNet stds
    }
}


class ThreadDetector:
    """
    Thread detector model for identifying threads in textile images.
    
    This implementation provides both traditional image processing techniques
    and a deep learning approach using UNet when available. The class automatically
    selects the best available method based on configuration and available dependencies.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True, 
                config_path: Optional[str] = None):
        """
        Initialize the thread detector.
        
        Args:
            model_path: Path to pre-trained UNet model weights.
                        If None, uses traditional image processing instead.
            use_gpu: Whether to use GPU acceleration for the neural network if available.
            config_path: Path to JSON configuration file with detection parameters.
                         If None, uses default configuration.
        """
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu and PYTORCH_AVAILABLE else "cpu")
        self.use_deep_learning = False
        self.model_path = model_path
        
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    # Update default config with custom values
                    for section in custom_config:
                        if section in self.config:
                            self.config[section].update(custom_config[section])
                    logger.info(f"Loaded custom configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
                logger.info("Using default configuration")
        
        # Set initial values
        self.model_type = "traditional"
        self.transforms = None
        
        # Initialize the model
        self._initialize_model()
        logger.info(f"Thread detector initialized using {self.model_type} approach")
    
    def _initialize_model(self) -> None:
        """
        Initialize the thread detection model based on available dependencies and configuration.
        This method will set up either a deep learning or traditional detection approach.
        """
        # Check if PyTorch is available and a model path is provided
        if PYTORCH_AVAILABLE and self.model_path and Path(self.model_path).exists():
            try:
                self._init_deep_learning_model()
                self.model_type = "unet"
                self.use_deep_learning = True
                logger.info(f"Initialized UNet model from {self.model_path} on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to load UNet model: {str(e)}. "
                              f"Falling back to traditional approach.")
                self.model_type = "traditional"
                self.use_deep_learning = False
        else:
            if not PYTORCH_AVAILABLE:
                logger.info("PyTorch not available. Using traditional image processing.")
            elif not self.model_path:
                logger.info("No model path provided. Using traditional image processing.")
            else:
                logger.warning(f"Model file not found: {self.model_path}. "
                              f"Using traditional image processing.")
            
            self.model_type = "traditional"
            self.use_deep_learning = False
    
    def _init_deep_learning_model(self) -> None:
        """
        Initialize the UNet model for deep learning based thread detection.
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning model")
            
        # Initialize UNet model
        self.model = self._create_unet_model()
        
        # Set up image transformations for preprocessing
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['image_processing']['normalize_mean'],
                std=self.config['image_processing']['normalize_std']
            )
        ])
        
        # Load pre-trained weights
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded model weights to {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load model weights: {str(e)}")
            
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Define image transformations for preprocessing
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def _create_unet_model(self):
        """
        Create a lightweight UNet model architecture for thread detection.
        
        Returns:
            PyTorch UNet model
        """
        if not PYTORCH_AVAILABLE:
            return None
            
        # This is a simplified UNet implementation
        # In production, use a more robust implementation with skip connections
        class UNetModel(nn.Module):
            def __init__(self):
                super(UNetModel, self).__init__()
                # Encoder
                self.enc1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True)
                )
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.enc2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True)
                )
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # Bridge
                self.bridge = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                
                # Decoder
                self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                self.dec1 = nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True)
                )
                self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
                self.dec2 = nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True)
                )
                
                # Output
                self.out = nn.Sequential(
                    nn.Conv2d(16, 2, kernel_size=1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                # Encoder
                enc1 = self.enc1(x)
                p1 = self.pool1(enc1)
                enc2 = self.enc2(p1)
                p2 = self.pool2(enc2)
                
                # Bridge
                bridge = self.bridge(p2)
                
                # Decoder
                up1 = self.upconv1(bridge)
                dec1 = self.dec1(up1)
                up2 = self.upconv2(dec1)
                dec2 = self.dec2(up2)
                
                # Output - channel 0 for warp, channel 1 for weft
                return self.out(dec2)
                
        return UNetModel()
        
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect threads in the input image using either deep learning or traditional methods.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Tuple of (warp_probability_map, weft_probability_map)
        """
        logger.debug(f"Running thread detection using {self.model_type} approach")
        start_time = time.time()
        
        # Validate input image
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
            
        # Store original image dimensions
        original_shape = image.shape
        
        # Resize large images for efficiency
        max_dim = self.config["image_processing"]["resize_max_dimension"]
        if max(original_shape[0], original_shape[1]) > max_dim:
            # Calculate scale factor
            scale = max_dim / max(original_shape[0], original_shape[1])
            new_width = int(original_shape[1] * scale)
            new_height = int(original_shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {original_shape} to {image.shape} for processing")
            
        if len(image.shape) > 2 and image.shape[2] > 1:
            # Convert to grayscale if necessary
            logger.debug("Converting multi-channel image to grayscale")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply pre-processing for better thread detection
        image = self._preprocess_image(image)
        
        if self.use_deep_learning and PYTORCH_AVAILABLE:
            try:
                # Use the deep learning model
                warp_prob, weft_prob = self._predict_with_unet(image)
                
                # Resize probability maps back to original dimensions if needed
                if original_shape[0] != image.shape[0] or original_shape[1] != image.shape[1]:
                    warp_prob = cv2.resize(warp_prob, (original_shape[1], original_shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
                    weft_prob = cv2.resize(weft_prob, (original_shape[1], original_shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
                
                detection_time = time.time() - start_time
                logger.info(f"UNet thread detection completed in {detection_time:.3f}s")
                return warp_prob, weft_prob
                
            except Exception as e:
                logger.error(f"Error in UNet prediction: {str(e)}. Falling back to traditional method.")
                
        # If UNet fails or not available, use traditional method
        logger.debug("Using traditional image processing for thread detection")
        
        # Detect horizontal threads (weft) using improved Sobel filtering
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        weft_prob = cv2.normalize(np.abs(grad_x), None, 0, 1, cv2.NORM_MINMAX)
        
        # Enhance weft probability using morphological operations
        kernel_weft = np.ones((1, 3), np.uint8)  # Horizontal kernel
        weft_prob = cv2.morphologyEx(weft_prob, cv2.MORPH_OPEN, kernel_weft)
        weft_prob = cv2.morphologyEx(weft_prob, cv2.MORPH_CLOSE, kernel_weft)
        
        # Detect vertical threads (warp) using improved Sobel filtering
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        warp_prob = cv2.normalize(np.abs(grad_y), None, 0, 1, cv2.NORM_MINMAX)
        
        # Enhance warp probability using morphological operations
        kernel_warp = np.ones((3, 1), np.uint8)  # Vertical kernel
        warp_prob = cv2.morphologyEx(warp_prob, cv2.MORPH_OPEN, kernel_warp)
        warp_prob = cv2.morphologyEx(warp_prob, cv2.MORPH_CLOSE, kernel_warp)
        
        detection_time = time.time() - start_time
        logger.debug(f"Thread detection completed in {detection_time:.3f}s")
        
        return warp_prob, weft_prob
        
    def _predict_with_unet(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict thread probability maps using the UNet model.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Tuple of (warp_probability_map, weft_probability_map)
        """
        if not PYTORCH_AVAILABLE or self.model is None:
            raise RuntimeError("PyTorch or UNet model not available")
            
        # Resize image to expected input size
        h, w = image.shape
        image_resized = cv2.resize(image, (256, 256))
        
        # Preprocess image
        image_tensor = self.transforms(image_resized).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(image_tensor)
            
        # Extract probability maps (channel 0 = warp, channel 1 = weft)
        warp_prob = output[0, 0].cpu().numpy()
        weft_prob = output[0, 1].cpu().numpy()
        
        # Resize back to original dimensions
        warp_prob = cv2.resize(warp_prob, (w, h))
        weft_prob = cv2.resize(weft_prob, (w, h))
        
        return warp_prob, weft_prob
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to enhance thread visibility.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image optimized for thread detection
        """
        try:
            # Apply Gaussian blur to reduce noise
            kernel_size = self.config["image_processing"]["gaussian_blur_kernel"]
            blurred = cv2.GaussianBlur(image, kernel_size, 0)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This enhances local contrast which makes threads more visible
            clahe = cv2.createCLAHE(
                clipLimit=self.config["image_processing"]["clahe_clip_limit"],
                tileGridSize=self.config["image_processing"]["clahe_grid_size"]
            )
            enhanced = clahe.apply(blurred)
            
            # Apply image sharpening to enhance edges
            kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            # If preprocessing fails, return the original image
            return image
    
    def post_process(
        self, 
        warp_prob: np.ndarray, 
        weft_prob: np.ndarray, 
        threshold: Optional[float] = None,
        min_thread_distance: Optional[int] = None,
        use_hough: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Process probability maps to extract thread positions using advanced techniques.
        
        Args:
            warp_prob: Probability map for warp threads
            weft_prob: Probability map for weft threads
            threshold: Threshold for binarizing probability maps (0.0-1.0)
            min_thread_distance: Minimum distance between adjacent threads
            use_hough: Whether to use Hough transform for line detection
            
        Returns:
            Tuple of (warp_thread_positions, weft_thread_positions)
        """
        # Use config values if parameters not specified
        if threshold is None:
            threshold = self.config["detection"]["threshold"]
        
        if min_thread_distance is None:
            min_thread_distance = self.config["detection"]["peak_distance"]
        
        # Input validation
        if warp_prob is None or weft_prob is None:
            raise ValueError("Probability maps cannot be None")
        
        if not 0.0 <= threshold <= 1.0:
            logger.warning(f"Invalid threshold value {threshold}, using default 0.5")
            threshold = 0.5
            
        logger.debug(f"Post-processing thread probability maps, threshold={threshold}, "
                   f"min_distance={min_thread_distance}, use_hough={use_hough}")
        
        try:
            # Binarize probability maps
            warp_binary = (warp_prob > threshold).astype(np.uint8) * 255
            weft_binary = (weft_prob > threshold).astype(np.uint8) * 255
            
            # Apply morphological operations to clean up binary images
            kernel_close = np.ones((3, 3), np.uint8)
            warp_binary = cv2.morphologyEx(warp_binary, cv2.MORPH_CLOSE, kernel_close)
            weft_binary = cv2.morphologyEx(weft_binary, cv2.MORPH_CLOSE, kernel_close)
            
            # Thin lines to single pixel width for better accuracy
            warp_binary = cv2.ximgproc.thinning(warp_binary) if hasattr(cv2, 'ximgproc') else warp_binary
            weft_binary = cv2.ximgproc.thinning(weft_binary) if hasattr(cv2, 'ximgproc') else weft_binary
        
            # Get thread positions using the appropriate method
            if use_hough:
                warp_positions = self._get_thread_positions_hough(
                    warp_binary, orientation='vertical', min_distance=min_thread_distance)
                weft_positions = self._get_thread_positions_hough(
                    weft_binary, orientation='horizontal', min_distance=min_thread_distance)
            else:
                warp_positions = self._get_thread_positions_projection(
                    warp_binary, axis=0, min_distance=min_thread_distance)
                weft_positions = self._get_thread_positions_projection(
                    weft_binary, axis=1, min_distance=min_thread_distance)
            
            logger.info(f"Thread detection complete: {len(warp_positions)} warp threads, "
                      f"{len(weft_positions)} weft threads")
                      
            return warp_positions, weft_positions
            
        except Exception as e:
            logger.error(f"Error in thread position extraction: {str(e)}", exc_info=True)
            # Return empty lists as fallback
            return [], []
        
        return warp_positions, weft_positions
    
    def _get_thread_positions_projection(
        self, 
        binary_map: np.ndarray, 
        axis: int,
        min_distance: int = 5,
        prominence: float = 0.3
    ) -> List[int]:
        """
        Extract thread positions from binary map using projection profile analysis.
        
        Args:
            binary_map: Binary image with threads
            axis: 0 for horizontal projection (warp threads), 1 for vertical projection (weft threads)
            min_distance: Minimum distance between adjacent threads
            prominence: Minimum height of peaks relative to surrounding values
            
        Returns:
            List of thread positions (sorted)
        """
        # Project along specified axis
        projection = np.sum(binary_map, axis=axis)
        
        # Apply Savitzky-Golay filter to smooth the projection
        try:
            window_length = min(21, len(projection) // 5 * 2 + 1)  # Must be odd
            if window_length >= 3:
                projection = signal.savgol_filter(projection, window_length, 3)
        except Exception as e:
            logger.warning(f"Failed to apply smoothing filter: {str(e)}")
        
        # Normalize projection
        if np.max(projection) > 0:
            projection = projection / np.max(projection)
        
        # Find peaks using scipy's find_peaks with advanced parameters
        try:
            peaks, _ = signal.find_peaks(
                projection,
                distance=min_distance,
                prominence=prominence
            )
            return sorted(peaks.tolist())
        except Exception as e:
            logger.error(f"Error finding peaks: {str(e)}")
            
            # Fall back to simple peak detection if scipy method fails
            positions = []
            window_size = min_distance
            
            for i in range(window_size, len(projection) - window_size):
                window = projection[i-window_size:i+window_size+1]
                if projection[i] == np.max(window) and projection[i] > prominence:
                    positions.append(i)
            
            return sorted(positions)
            
    def _get_thread_positions_hough(
        self,
        binary_map: np.ndarray,
        orientation: str = 'vertical',
        min_distance: int = 5,
        min_line_length: int = 30,
        max_line_gap: int = 10
    ) -> List[int]:
        """
        Extract thread positions using Hough Transform line detection.
        
        Args:
            binary_map: Binary image with threads
            orientation: 'vertical' for warp threads, 'horizontal' for weft threads
            min_distance: Minimum distance between detected lines
            min_line_length: Minimum length of line
            max_line_gap: Maximum allowed gap in a line
            
        Returns:
            List of thread positions (sorted)
        """
        # Set angle range based on orientation
        if orientation == 'vertical':
            theta_range = np.array([np.pi/2]) # Vertical lines
        else:
            theta_range = np.array([0])  # Horizontal lines
            
        # Apply Hough Transform
        try:
            # Standard Hough Line Transform
            lines = cv2.HoughLines(
                binary_map, 
                rho=1, 
                theta=np.pi/180, 
                threshold=binary_map.shape[0]//4,  # Adjust based on image size
                min_theta=theta_range[0] - np.pi/12,  # Allow some deviation
                max_theta=theta_range[0] + np.pi/12
            )
            
            if lines is None:
                logger.warning(f"No lines detected with standard Hough transform, trying probabilistic")
                return self._get_thread_positions_projection(
                    binary_map, 
                    axis=0 if orientation == 'vertical' else 1, 
                    min_distance=min_distance
                )
                
            # Extract positions from lines
            positions = []
            for line in lines:
                rho, theta = line[0]
                if orientation == 'vertical':
                    # For vertical lines, rho is approximately the x-position
                    x = int(np.abs(rho))
                    positions.append(x)
                else:
                    # For horizontal lines, rho is approximately the y-position
                    y = int(np.abs(rho))
                    positions.append(y)
                    
            # Remove duplicates by clustering close lines
            positions = sorted(positions)
            if len(positions) > 1:
                clustered_positions = [positions[0]]
                for pos in positions[1:]:
                    if pos - clustered_positions[-1] >= min_distance:
                        clustered_positions.append(pos)
                positions = clustered_positions
            
            return positions
            
        except Exception as e:
            logger.error(f"Error in Hough line detection: {str(e)}")
            # Fall back to projection method
            return self._get_thread_positions_projection(
                binary_map, 
                axis=0 if orientation == 'vertical' else 1,
                min_distance=min_distance
            )
