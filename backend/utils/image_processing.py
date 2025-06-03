"""
Image processing utilities for thread analysis.

This module provides optimized image processing functions for preprocessing,
enhancing, and analyzing textile images for thread detection.
"""
import os
import cv2
import numpy as np
from scipy import ndimage, signal
import math
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
import time
import uuid

from utils.logging_config import get_logger

# Initialize logger
logger = get_logger("utils.image_processing")

# Default configuration values
DEFAULT_CONFIG = {
    "adaptive_threshold": {
        "block_size": 11,
        "c_value": 2
    },
    "canny_edge": {
        "low_threshold": 50,
        "high_threshold": 150,
        "aperture_size": 3
    },
    "gabor_filter": {
        "ksize": (15, 15),
        "sigma": 4.0,
        "theta": 0,  # Will be dynamically adjusted
        "lambda_val": 10.0,
        "gamma": 0.5,
        "psi": 0
    },
    "clahe": {
        "clip_limit": 2.0,
        "tile_grid_size": (8, 8)
    },
    "orientation": {
        "angle_threshold": 5.0  # Degrees from horizontal/vertical to trigger rotation
    }
}


class PreprocessingMethod(Enum):
    """Enumeration of available preprocessing methods."""
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    OTSU_THRESHOLD = "otsu_threshold"
    CANNY_EDGE = "canny_edge"
    GABOR_FILTER = "gabor_filter"
    MULTI_SCALE = "multi_scale"  # Combines multiple methods


class OrientationCorrection(Enum):
    """Enumeration of orientation correction methods."""
    NONE = "none"
    AUTO = "auto"
    FORCED_HORIZONTAL = "horizontal"
    FORCED_VERTICAL = "vertical"
    MANUAL = "manual"
    
    
class ImageProcessingError(Exception):
    """Exception raised for errors in image processing operations."""
    pass


def validate_image(image: np.ndarray, min_dimensions: Tuple[int, int] = (50, 50), 
                 max_dimensions: Tuple[int, int] = (10000, 10000)) -> bool:
    """
    Validate if the provided array is a valid image with acceptable dimensions.
    
    Args:
        image: Input image array
        min_dimensions: Minimum acceptable dimensions (width, height)
        max_dimensions: Maximum acceptable dimensions (width, height)
        
    Returns:
        True if valid image, False otherwise
        
    Raises:
        ImageProcessingError: With detailed error message if validation fails
    """
    try:
        if image is None:
            raise ImageProcessingError("Image is None")
        
        if not isinstance(image, np.ndarray):
            raise ImageProcessingError(f"Expected numpy array, got {type(image)}")
            
        if len(image.shape) < 2:
            raise ImageProcessingError(f"Invalid image dimensions: {image.shape}")
        
        if len(image.shape) > 2 and image.shape[2] > 4:
            raise ImageProcessingError(f"Too many channels: {image.shape[2]}")
            
        height, width = image.shape[:2]
        
        if width < min_dimensions[0] or height < min_dimensions[1]:
            raise ImageProcessingError(
                f"Image too small: {width}x{height}, minimum: {min_dimensions[0]}x{min_dimensions[1]}"
            )
            
        if width > max_dimensions[0] or height > max_dimensions[1]:
            raise ImageProcessingError(
                f"Image too large: {width}x{height}, maximum: {max_dimensions[0]}x{max_dimensions[1]}"
            )
            
        return True
            
    except ImageProcessingError as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error validating image: {str(e)}")
        raise ImageProcessingError(f"Validation failed: {str(e)}")
        
def preprocess_image(
    image: np.ndarray, 
    method: Union[str, PreprocessingMethod] = PreprocessingMethod.ADAPTIVE_THRESHOLD,
    denoise_strength: int = 5,
    correct_orientation: Union[str, OrientationCorrection] = OrientationCorrection.AUTO,
    orientation_angle: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess image for thread detection with multiple enhancement options.
    
    Args:
        image: Input image
        method: Preprocessing method to use
        denoise_strength: Strength of the denoising (odd number)
        correct_orientation: Method to correct image orientation
        orientation_angle: Manual orientation angle correction in degrees (if mode is MANUAL)
        
    Returns:
        Tuple of (preprocessed_image, metadata_dict)
    """
    start_time = time.time()
    
    # Validate input
    if not validate_image(image):
        raise ValueError("Invalid input image")
    
    # Ensure denoise_strength is odd
    if denoise_strength % 2 == 0:
        denoise_strength += 1
    
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Correct orientation if needed
    corrected_image, angle = _correct_orientation(
        gray, 
        mode=correct_orientation, 
        manual_angle=orientation_angle
    )
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(corrected_image, (denoise_strength, denoise_strength), 0)
    
    # Process image based on selected method
    if isinstance(method, str):
        try:
            method = PreprocessingMethod(method)
        except ValueError:
            logger.warning(f"Invalid preprocessing method: {method}. Using default.")
            method = PreprocessingMethod.ADAPTIVE_THRESHOLD
    
    # Apply selected preprocessing method
    if method == PreprocessingMethod.ADAPTIVE_THRESHOLD:
        processed = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == PreprocessingMethod.OTSU_THRESHOLD:
        _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == PreprocessingMethod.CANNY_EDGE:
        processed = cv2.Canny(blurred, 50, 150)
    elif method == PreprocessingMethod.GABOR_FILTER:
        processed = _apply_gabor_filter(blurred)
    else:
        # Default to adaptive threshold
        processed = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    
    # Record metadata
    metadata = {
        "preprocessing_method": method.value if isinstance(method, PreprocessingMethod) else str(method),
        "orientation_correction": {
            "applied": correct_orientation != OrientationCorrection.NONE,
            "angle": angle,
            "method": correct_orientation.value if isinstance(correct_orientation, OrientationCorrection) else str(correct_orientation)
        },
        "denoise_strength": denoise_strength,
        "processing_time": time.time() - start_time
    }
    
    logger.debug(f"Preprocessing completed in {metadata['processing_time']:.3f}s using {metadata['preprocessing_method']}")
    
    return processed, metadata


def _apply_gabor_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply Gabor filter for texture analysis to highlight thread patterns.
    
    Args:
        image: Grayscale input image
        
    Returns:
        Filtered image
    """
    # Define Gabor filter parameters for horizontal and vertical threads
    ksize = 31  # Filter size
    sigma = 4.0  # Standard deviation
    theta_h = 0  # Horizontal orientation (0 degrees)
    theta_v = np.pi/2  # Vertical orientation (90 degrees)
    lambd = 10.0  # Wavelength
    gamma = 0.5  # Aspect ratio
    psi = 0  # Phase offset
    
    # Apply horizontal Gabor filter
    g_kernel_h = cv2.getGaborKernel((ksize, ksize), sigma, theta_h, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_h = cv2.filter2D(image, cv2.CV_8UC3, g_kernel_h)
    
    # Apply vertical Gabor filter
    g_kernel_v = cv2.getGaborKernel((ksize, ksize), sigma, theta_v, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_v = cv2.filter2D(image, cv2.CV_8UC3, g_kernel_v)
    
    # Combine horizontal and vertical responses
    combined = cv2.addWeighted(filtered_h, 0.5, filtered_v, 0.5, 0)
    
    # Apply threshold to create binary image
    _, binary = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
    
    return binary


def _correct_orientation(
    image: np.ndarray, 
    mode: Union[str, OrientationCorrection] = OrientationCorrection.AUTO,
    manual_angle: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Correct the orientation of the fabric image to ensure threads are aligned with axes.
    
    Args:
        image: Input grayscale image
        mode: Orientation correction mode
        manual_angle: Manual angle correction in degrees (if mode is MANUAL)
        
    Returns:
        Tuple of (corrected_image, rotation_angle)
        
    Raises:
        ImageProcessingError: If orientation correction fails
    """
    try:
        # Validate input
        validate_image(image)
        
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = OrientationCorrection(mode)
            except ValueError:
                logger.warning(f"Invalid orientation mode: {mode}. Using AUTO.")
                mode = OrientationCorrection.AUTO
        
        # Skip correction if NONE
        if mode == OrientationCorrection.NONE:
            return image, 0.0
            
        # Apply manual rotation if specified
        if mode == OrientationCorrection.MANUAL:
            if manual_angle is not None:
                angle = manual_angle
            else:
                logger.warning("Manual orientation requested but no angle provided. Using 0.")
                angle = 0.0
        
        # Auto-detect orientation using Hough transform
        elif mode == OrientationCorrection.AUTO:
            angle = _detect_orientation(image)
        
        # Force horizontal or vertical alignment
        elif mode == OrientationCorrection.FORCED_HORIZONTAL:
            detected_angle = _detect_orientation(image)
            # Align closer to horizontal (0 or 180 degrees)
            angle = detected_angle % 90
            if angle > 45:
                angle -= 90
        
        elif mode == OrientationCorrection.FORCED_VERTICAL:
            detected_angle = _detect_orientation(image)
            # Align closer to vertical (90 or 270 degrees)
            angle = detected_angle % 90
            if angle < 45:
                angle += 90
        
        # Apply rotation correction if needed
        if abs(angle) > 1.0:  # Only correct if angle is significant
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            corrected = cv2.warpAffine(image, M, (cols, rows), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
            logger.debug(f"Image orientation corrected by {angle:.2f} degrees")
            return corrected, angle
        
        # No significant rotation needed
        return image, 0.0
    
    except cv2.error as cv_err:
        logger.error(f"OpenCV error during orientation correction: {str(cv_err)}")
        raise ImageProcessingError(f"Orientation correction failed: {str(cv_err)}")
    except Exception as e:
        logger.error(f"Unexpected error during orientation correction: {str(e)}")
        # Return original image with zero angle if error occurs
        logger.warning("Returning original image due to orientation correction failure")
        return image, 0.0


def _detect_orientation(image: np.ndarray) -> float:
    """
    Detect the orientation of threads in the image using Hough transform.
    
    Args:
        image: Grayscale input image
        
    Returns:
        Detected angle in degrees
    """
    # Detect edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Apply Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(min(image.shape) / 2))
    
    if lines is None or len(lines) == 0:
        logger.warning("No lines detected for orientation correction")
        return 0.0
        
    # Collect angles
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert to degrees in range [-90, 90]
        angle_deg = math.degrees(theta) - 90
        angles.append(angle_deg % 180 - 90)
        
    # Find the dominant angle using a histogram approach
    hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
    dominant_bin = np.argmax(hist)
    dominant_angle = (bins[dominant_bin] + bins[dominant_bin + 1]) / 2
    
    logger.debug(f"Detected fabric orientation: {dominant_angle:.2f} degrees")
    
    return dominant_angle


def enhance_contrast(
    image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    grid_size: int = 8,
    brightness_adjustment: float = 0.0,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enhance image contrast for better thread detection with multiple techniques.
    
    Args:
        image: Grayscale input image
        method: Enhancement method ("clahe", "histogram_eq", "adaptive", "multi")
        clip_limit: Clip limit for CLAHE (1.0-5.0)
        grid_size: Grid size for CLAHE (4-16)
        brightness_adjustment: Adjust brightness (-1.0 to 1.0)
        config: Optional configuration dictionary overriding defaults
        
    Returns:
        Tuple of (enhanced_image, metadata_dict)
        
    Raises:
        ImageProcessingError: If enhancement fails
    """
    start_time = time.time()
    
    try:
        # Validate input
        validate_image(image)
        
        # Ensure grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply median blur to reduce noise while preserving edges
        denoised = cv2.medianBlur(gray, 3)
        
        # Initialize metadata
        metadata = {
            "method": method,
            "parameters": {}
        }
        
            # Apply enhancement based on selected method
        if method.lower() == "clahe":
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            enhanced = clahe.apply(denoised)
            metadata["parameters"] = {"clip_limit": clip_limit, "grid_size": grid_size}
            
        elif method.lower() == "histogram_eq":
            # Apply standard histogram equalization
            enhanced = cv2.equalizeHist(denoised)
            
        elif method.lower() == "adaptive":
            # Apply local adaptive histogram equalization for challenging lighting
            # First apply bilateral filter to preserve edges
            bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
            
            # Then apply CLAHE with moderate parameters
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            enhanced = clahe.apply(bilateral)
            metadata["parameters"] = {"clip_limit": clip_limit, "grid_size": grid_size, "bilateral_filter": True}
            
        elif method.lower() == "multi":
            # Advanced multi-technique method for challenging fabrics
            # Step 1: Apply bilateral filter to smooth while preserving edges
            bilateral = cv2.bilateralFilter(denoised, 7, 50, 50)
            
            # Step 2: Apply CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            clahe_result = clahe.apply(bilateral)
            
            # Step 3: Create a sharpened version
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(clahe_result, -1, kernel)
            
            # Step 4: Apply adaptive thresholding for improved local contrast
            adapt_threshold = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Step 5: Blend with original using weighted average for best results
            enhanced = cv2.addWeighted(clahe_result, 0.7, adapt_threshold, 0.3, 0)
            
            metadata["parameters"] = {
                "clip_limit": clip_limit, 
                "grid_size": grid_size,
                "sharpening": True,
                "adaptive_threshold": True,
                "blend_ratio": [0.7, 0.3]
            }
        
        else:
            # Default to CLAHE
            logger.warning(f"Unknown enhancement method: {method}. Using CLAHE.")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            method = "clahe"
            metadata["parameters"] = {"clip_limit": 2.0, "grid_size": 8}
        
        # Apply brightness adjustment if requested
        if brightness_adjustment != 0.0:
            brightness = int(brightness_adjustment * 100)  # Scale to pixel values
            if brightness > 0:
                enhanced = cv2.add(enhanced, np.ones(enhanced.shape, dtype="uint8") * brightness)
            else:
                enhanced = cv2.subtract(enhanced, np.ones(enhanced.shape, dtype="uint8") * abs(brightness))
            metadata["parameters"]["brightness_adjustment"] = brightness_adjustment
            
        # Apply additional Gaussian blur if the image is very noisy
        if np.std(enhanced) > 75:  # High standard deviation indicates noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
            metadata["parameters"]["additional_blur"] = True
            
        metadata["processing_time"] = time.time() - start_time
        logger.debug(f"Contrast enhancement completed in {metadata['processing_time']:.3f}s using {method}")
        
        return enhanced, metadata
        
    except cv2.error as cv_err:
        logger.error(f"OpenCV error during contrast enhancement: {str(cv_err)}")
        raise ImageProcessingError(f"Failed to enhance contrast: {str(cv_err)}")
    except Exception as e:
        logger.error(f"Unexpected error during contrast enhancement: {str(e)}")
        raise ImageProcessingError(f"Contrast enhancement failed: {str(e)}")


def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150, 
                aperture_size: int = 3) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detection with configurable parameters.
    
    Args:
        image: Preprocessed grayscale image
        low_threshold: Lower threshold for hysteresis procedure
        high_threshold: Higher threshold for hysteresis procedure
        aperture_size: Aperture size for Sobel operator
        
    Returns:
        Edge image
        
    Raises:
        ImageProcessingError: If edge detection fails
    """
    try:
        # Validate input
        validate_image(image)
        
        # Ensure grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)
        
        return edges
        
    except cv2.error as cv_err:
        logger.error(f"OpenCV error during edge detection: {str(cv_err)}")
        raise ImageProcessingError(f"Edge detection failed: {str(cv_err)}")
    except Exception as e:
        logger.error(f"Unexpected error during edge detection: {str(e)}")
        raise ImageProcessingError(f"Edge detection failed: {str(e)}")


def resize_image(image: np.ndarray, max_dimension: int = 1024, 
                interpolation: int = cv2.INTER_AREA) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio with proper error handling.
    
    Args:
        image: Input image
        max_dimension: Maximum dimension (width or height)
        interpolation: Interpolation method (cv2.INTER_AREA recommended for downscaling)
        
    Returns:
        Tuple of (resized_image, scale_factor)
        
    Raises:
        ImageProcessingError: If resizing fails
    """
    try:
        # Validate input
        validate_image(image)
        
        height, width = image.shape[:2]
        scale_factor = 1.0
        
        # Calculate scale factor
        if height > max_dimension or width > max_dimension:
            if height > width:
                scale_factor = max_dimension / height
            else:
                scale_factor = max_dimension / width
        
        # Resize if necessary
        if scale_factor != 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            logger.debug(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            return resized, scale_factor
        
        return image, scale_factor
        
    except cv2.error as cv_err:
        logger.error(f"OpenCV error during image resizing: {str(cv_err)}")
        raise ImageProcessingError(f"Image resizing failed: {str(cv_err)}")
    except Exception as e:
        logger.error(f"Unexpected error during image resizing: {str(e)}")
        raise ImageProcessingError(f"Image resizing failed: {str(e)}")
