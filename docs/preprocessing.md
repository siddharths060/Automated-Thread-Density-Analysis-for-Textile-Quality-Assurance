# Image Preprocessing Pipeline

## Overview

The preprocessing pipeline plays a crucial role in preparing fabric images for thread detection and counting. A well-designed preprocessing workflow enhances thread visibility, reduces noise, and normalizes variations in lighting and capture conditions.

![Preprocessing Pipeline](assets/preprocessing-pipeline.png)

## Preprocessing Steps

### 1. Image Resizing Strategy

#### DPI Normalization
- **Purpose**: Ensure consistent pixel-to-inch ratio for accurate thread counting
- **Implementation**: 
  - Target resolution of 300 DPI (dots per inch)
  - If image contains DPI metadata, use it to rescale
  - If no metadata, use calibration markers or assume standard 96 DPI

```python
def normalize_dpi(image, target_dpi=300):
    """Normalize image to target DPI for consistent thread counting"""
    # Get image metadata
    metadata = image.info.get('dpi', (96, 96))
    original_dpi = metadata[0]
    
    # Calculate scaling factor
    scale_factor = target_dpi / original_dpi
    
    # Resize image
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    return image.resize((new_width, new_height), Image.LANCZOS)
```

#### Fixed Resolution
- For deep learning model input, resize to fixed dimensions (512x512 pixels)
- Maintain aspect ratio with padding when necessary
- Use high-quality interpolation (Lanczos for downsampling, bicubic for upsampling)

### 2. Grayscale Conversion

- **Purpose**: Reduce dimensionality and focus on intensity patterns
- **Implementation**: 
  - Convert RGB to grayscale using weighted channel combination
  - Y = 0.299R + 0.587G + 0.114B (ITU-R BT.601 standard)
  - For specific fabric types, custom channel weighting may be applied

```python
def convert_to_grayscale(image):
    """Convert image to grayscale with appropriate channel weights"""
    if len(image.shape) == 3:  # Color image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already grayscale
```

### 3. Noise Removal Techniques

#### Gaussian Blur
- **Purpose**: Remove high-frequency noise
- **Parameters**: 
  - Kernel size: 3×3 or 5×5
  - Sigma: 0.5 to 1.0
  - Preserves edges while smoothing noise

#### Median Filter
- **Purpose**: Remove salt-and-pepper noise while preserving edges
- **Parameters**:
  - Kernel size: 3×3 or 5×5
  - Particularly effective for digital camera artifacts

```python
def remove_noise(image, method="gaussian", kernel_size=5):
    """Apply noise reduction filters to the image"""
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0.8)
    elif method == "median":
        return cv2.medianBlur(image, kernel_size)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image
```

### 4. Edge Detection

#### Canny Edge Detection
- **Purpose**: Identify thread boundaries
- **Parameters**:
  - Lower threshold: 50
  - Upper threshold: 150
  - Aperture size: 3

#### Sobel Operator
- **Purpose**: Detect horizontal and vertical edges separately
- **Implementation**:
  - Horizontal kernel (dx=1, dy=0) for weft threads
  - Vertical kernel (dx=0, dy=1) for warp threads
  - Kernel size: 3×3 or 5×5

```python
def detect_edges(image, method="canny"):
    """Detect edges in the image to highlight thread boundaries"""
    if method == "canny":
        return cv2.Canny(image, 50, 150, L2gradient=True)
    elif method == "sobel":
        # Combine gradients in x and y directions
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return image
```

### 5. Morphological Operations

#### Dilation
- **Purpose**: Thicken and connect broken thread lines
- **Parameters**:
  - Kernel shape: Rectangular or elliptical
  - Kernel size: 3×3 or 5×5
  - Iterations: 1 or 2

#### Erosion
- **Purpose**: Thin threads and separate close threads
- **Parameters**:
  - Kernel shape: Rectangular or elliptical
  - Kernel size: 3×3
  - Iterations: 1

#### Opening/Closing
- **Purpose**: Remove small noise objects and fill small holes
- **Implementation**:
  - Opening = Erosion followed by dilation
  - Closing = Dilation followed by erosion

```python
def apply_morphology(image, operation, kernel_size=3, iterations=1):
    """Apply morphological operations to enhance thread patterns"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == "dilate":
        return cv2.dilate(image, kernel, iterations=iterations)
    elif operation == "erode":
        return cv2.erode(image, kernel, iterations=iterations)
    elif operation == "open":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "close":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return image
```

### 6. Binarization and Thresholding

#### Adaptive Thresholding
- **Purpose**: Convert grayscale to binary image, enhancing thread contrast
- **Implementation**:
  - Block size: 11×11 or 15×15
  - Constant subtraction: 2 to 5
  - Adapts to local lighting conditions

#### Otsu's Method
- **Purpose**: Automatic threshold determination
- **Implementation**:
  - Bimodal histogram analysis
  - Maximizes variance between thread and background classes

```python
def apply_threshold(image, method="adaptive"):
    """Apply thresholding to separate threads from background"""
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    elif method == "otsu":
        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh
    return image
```

### 7. ROI (Region of Interest) Cropping

#### Static ROI Selection
- **Purpose**: Focus on a specific fabric area for analysis
- **Implementation**:
  - Center crop with configurable dimensions (e.g., 1×1 inch)
  - Padding or border extension when necessary

#### Dynamic ROI Detection
- **Purpose**: Automatically identify suitable areas for thread counting
- **Implementation**:
  - Texture homogeneity analysis
  - Avoid areas with defects, labels, or uneven texture
  - Select multiple ROIs for validation and averaging

```python
def crop_roi(image, method="center", size_inches=1):
    """Crop region of interest for thread counting"""
    h, w = image.shape[:2]
    
    if method == "center":
        # Convert inches to pixels (assuming 300 DPI)
        size_pixels = int(size_inches * 300)
        
        # Calculate center crop coordinates
        center_x, center_y = w // 2, h // 2
        x1 = max(0, center_x - size_pixels // 2)
        y1 = max(0, center_y - size_pixels // 2)
        x2 = min(w, x1 + size_pixels)
        y2 = min(h, y1 + size_pixels)
        
        return image[y1:y2, x1:x2]
    
    # Add other ROI selection methods here
    return image
```

## Complete Preprocessing Pipeline

The following function combines all preprocessing steps into a complete pipeline:

```python
def preprocess_fabric_image(image_path, params=None):
    """
    Complete preprocessing pipeline for fabric images
    
    Args:
        image_path (str): Path to the input image
        params (dict): Optional parameters for each step
        
    Returns:
        dict: Processed images at each stage
    """
    # Default parameters
    if params is None:
        params = {
            "resize_dpi": 300,
            "noise_method": "gaussian",
            "noise_kernel": 5,
            "edge_method": "canny",
            "morph_operation": "close",
            "morph_kernel": 3,
            "threshold_method": "adaptive",
            "roi_method": "center",
            "roi_size": 1
        }
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Create result dictionary to store intermediate results
    result = {"original": image.copy()}
    
    # Resize to target DPI
    # (Assuming a function to extract DPI from metadata)
    image = resize_to_dpi(image, params["resize_dpi"])
    result["resized"] = image.copy()
    
    # Convert to grayscale
    gray = convert_to_grayscale(image)
    result["grayscale"] = gray.copy()
    
    # Remove noise
    denoised = remove_noise(
        gray, method=params["noise_method"], 
        kernel_size=params["noise_kernel"]
    )
    result["denoised"] = denoised.copy()
    
    # Detect edges
    edges = detect_edges(denoised, method=params["edge_method"])
    result["edges"] = edges.copy()
    
    # Apply morphological operations
    morph = apply_morphology(
        edges, 
        operation=params["morph_operation"],
        kernel_size=params["morph_kernel"]
    )
    result["morphology"] = morph.copy()
    
    # Apply thresholding
    thresh = apply_threshold(morph, method=params["threshold_method"])
    result["threshold"] = thresh.copy()
    
    # Crop ROI
    roi = crop_roi(
        thresh, 
        method=params["roi_method"], 
        size_inches=params["roi_size"]
    )
    result["roi"] = roi.copy()
    
    return result
```

## Parameter Tuning for Different Fabric Types

Different fabric types require different preprocessing parameters:

| Fabric Type | Noise Reduction | Edge Detection | Morphology | Thresholding |
|-------------|-----------------|----------------|------------|--------------|
| Cotton      | Gaussian, k=5   | Canny          | Close, k=3 | Adaptive     |
| Silk        | Bilateral       | Sobel          | Open, k=3  | Otsu         |
| Polyester   | Median, k=3     | Canny          | Close, k=5 | Adaptive     |
| Wool        | Gaussian, k=7   | Sobel          | Open, k=5  | Otsu         |
| Denim       | Median, k=5     | Sobel          | Close, k=7 | Adaptive     |

## Visual Examples of Preprocessing Steps

![Preprocessing Steps Example](assets/preprocessing-steps.png)

The image above shows each step of the preprocessing pipeline applied to a cotton fabric sample:
1. Original image
2. Grayscale conversion
3. Noise removal
4. Edge detection
5. Morphological operations
6. Thresholding
7. Final ROI with detected threads

## Conclusion

A robust preprocessing pipeline is essential for accurate thread detection and counting. The steps outlined above enhance thread visibility, normalize variations in image capture conditions, and prepare the image for either traditional computer vision algorithms or deep learning model inference.

The pipeline parameters can be adjusted based on fabric type, image quality, and specific requirements, making the system adaptable to various textile quality assurance applications.
