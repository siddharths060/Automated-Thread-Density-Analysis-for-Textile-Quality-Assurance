# ⚙️ Backend Implementation

This document details the FastAPI backend implementation for the Automated Thread Density Analysis system, covering the server architecture, API endpoints, image processing pipeline, and ML model integration.

## Architecture Overview

The backend is built using:
- **FastAPI**: High-performance Python web framework
- **OpenCV**: Computer vision for image processing
- **PyTorch**: Deep learning model inference
- **Pillow**: Additional image manipulation
- **NumPy/SciPy**: Numerical processing
- **SQLAlchemy**: Database interactions (optional)
- **Pydantic**: Data validation and settings management

![Backend Architecture](assets/backend-architecture.png)

## Project Structure

The backend follows a modular, layered architecture:

```
backend/
├── api/
│   ├── __init__.py
│   ├── endpoints/
│   │   ├── __init__.py
│   │   ├── upload.py
│   │   ├── predict.py
│   │   ├── results.py
│   │   └── status.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   └── responses.py
│   └── routes.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   └── errors.py
├── processing/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── thread_detection.py
│   └── thread_counting.py
├── ml/
│   ├── __init__.py
│   ├── model.py
│   ├── inference.py
│   └── postprocessing.py
├── storage/
│   ├── __init__.py
│   ├── file_storage.py
│   └── database.py
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   └── validation.py
├── main.py
├── requirements.txt
└── Dockerfile
```

## Main Application Entry Point

The `main.py` file serves as the entry point for the FastAPI application:

```python
# main.py
import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.errors import APIError
from api.routes import router as api_router
from ml.model import initialize_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="API for automated thread counting in textile fabrics",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs(settings.STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Include API router
app.include_router(api_router, prefix="/api")

# Error handler
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )

# Startup event to load ML model
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Thread Density Analysis API server")
    initialize_model()
    logger.info("ML model initialized successfully")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
```

## Configuration Management

The `config.py` file handles all application settings through Pydantic:

```python
# core/config.py
from pydantic import BaseSettings, AnyHttpUrl
from typing import List, Optional, Union
import os
from pathlib import Path

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Thread Density Analysis API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Directory settings
    PROJECT_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = PROJECT_DIR / "uploads"
    STATIC_DIR: Path = PROJECT_DIR / "static"
    RESULTS_DIR: Path = STATIC_DIR / "results"
    
    # API settings
    API_PREFIX: str = "/api"
    CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React dev server
        "https://thread-density-analyzer.com"  # Production frontend
    ]
    
    # ML model settings
    MODEL_PATH: Path = PROJECT_DIR / "ml/models/thread_detection_unet.pt"
    MODEL_INPUT_SIZE: tuple = (512, 512)
    
    # Storage settings
    MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "tiff"]
    
    # Processing settings
    DPI_DEFAULT: int = 300
    THREAD_MIN_LENGTH_PIXELS: int = 30
    
    class Config:
        case_sensitive = True

# Create and export settings instance
settings = Settings()

# Create required directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
```

## API Routes

The API routes are organized into separate modules for each feature:

```python
# api/routes.py
from fastapi import APIRouter
from api.endpoints import upload, predict, results, status

# Create main router
router = APIRouter()

# Include endpoint routers
router.include_router(upload.router, tags=["Upload"])
router.include_router(predict.router, tags=["Prediction"])
router.include_router(results.router, tags=["Results"])
router.include_router(status.router, tags=["Status"])
```

## Upload Endpoint

The upload endpoint handles image uploads and initial validation:

```python
# api/endpoints/upload.py
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from core.config import settings
from utils.validation import validate_image
from api.models.responses import UploadResponse
from storage.file_storage import save_uploaded_file

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    image: UploadFile = File(...),
    dpi: Optional[int] = Form(None),
    calibration_square_size: Optional[float] = Form(1.0)
):
    """
    Upload a fabric image for thread count analysis.
    
    - **image**: The fabric image file (JPEG, PNG, TIFF)
    - **dpi**: Image DPI if known (default: auto-detect)
    - **calibration_square_size**: Size in inches of calibration square if present
    """
    try:
        # Validate image format and size
        await validate_image(
            image, 
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_extensions=settings.ALLOWED_EXTENSIONS
        )
        
        # Generate unique image ID
        image_id = f"img_{uuid.uuid4().hex[:8]}"
        
        # Save uploaded file
        file_path = await save_uploaded_file(
            image,
            dest_dir=settings.UPLOAD_DIR,
            filename=f"{image_id}{os.path.splitext(image.filename)[1]}"
        )
        
        # Store metadata
        metadata = {
            "original_filename": image.filename,
            "content_type": image.content_type,
            "dpi": dpi or settings.DPI_DEFAULT,
            "calibration_square_size": calibration_square_size,
            "upload_timestamp": str(datetime.utcnow())
        }
        
        # Save metadata to file
        with open(os.path.join(settings.UPLOAD_DIR, f"{image_id}.json"), "w") as f:
            json.dump(metadata, f)
        
        return {
            "success": True,
            "image_id": image_id,
            "filename": image.filename,
            "upload_timestamp": metadata["upload_timestamp"],
            "message": "Image uploaded successfully. Use the image_id to request analysis."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Predict Endpoint

The prediction endpoint handles thread detection and counting:

```python
# api/endpoints/predict.py
import os
import asyncio
import json
from fastapi import APIRouter, BackgroundTasks, HTTPException, Body, Depends
from typing import Optional, Dict, Any

from core.config import settings
from core.errors import APIError
from api.models.requests import PredictionRequest
from api.models.responses import PredictionResponse
from processing.thread_detection import process_image
from ml.inference import run_inference
from storage.file_storage import get_image_path, save_results

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_thread_count(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Process an uploaded image and return thread count analysis.
    
    - **image_id**: ID of the previously uploaded image
    - **options**: Optional processing parameters
    """
    image_id = request.image_id
    options = request.options.dict() if request.options else {}
    
    # Check if image exists
    image_path = get_image_path(image_id)
    if not image_path:
        raise APIError(
            status_code=404, 
            error_code="IMAGE_NOT_FOUND",
            message=f"Image with ID {image_id} not found"
        )
    
    # Try to load metadata
    metadata_path = os.path.join(settings.UPLOAD_DIR, f"{image_id}.json")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {"dpi": settings.DPI_DEFAULT}
    
    # Create status file to track progress
    status_data = {
        "image_id": image_id,
        "status": "queued",
        "progress": 0,
        "message": "Analysis queued"
    }
    
    status_path = os.path.join(settings.UPLOAD_DIR, f"{image_id}_status.json")
    with open(status_path, "w") as f:
        json.dump(status_data, f)
    
    # Run processing in background task to avoid blocking
    background_tasks.add_task(
        _process_image_task,
        image_id=image_id,
        image_path=image_path,
        metadata=metadata,
        options=options,
        status_path=status_path
    )
    
    return {
        "success": True,
        "image_id": image_id,
        "message": "Image processing started. Check status endpoint for progress."
    }

async def _process_image_task(
    image_id: str,
    image_path: str,
    metadata: Dict[str, Any],
    options: Dict[str, Any],
    status_path: str
):
    """Background task to process image and detect threads."""
    try:
        # Update status to processing
        _update_status(status_path, "preprocessing", 10, "Preprocessing image")
        
        # Preprocess image
        preprocessed_image = process_image(
            image_path=image_path,
            dpi=metadata.get("dpi", settings.DPI_DEFAULT),
            **options
        )
        
        # Update status to model inference
        _update_status(status_path, "processing", 30, "Running thread detection model")
        
        # Run model inference
        thread_mask = run_inference(preprocessed_image)
        
        # Update status to thread counting
        _update_status(status_path, "processing", 60, "Counting threads")
        
        # Count threads and generate results
        results = analyze_thread_mask(
            thread_mask, 
            original_image_path=image_path,
            dpi=metadata.get("dpi", settings.DPI_DEFAULT),
            **options
        )
        
        # Save results
        _update_status(status_path, "processing", 90, "Saving results")
        save_results(image_id, results)
        
        # Update status to completed
        _update_status(status_path, "completed", 100, "Analysis completed")
        
    except Exception as e:
        # Update status to failed
        _update_status(
            status_path, 
            "failed", 
            0, 
            f"Analysis failed: {str(e)}"
        )
        raise

def _update_status(status_path: str, status: str, progress: int, message: str):
    """Helper to update processing status file."""
    with open(status_path, "r") as f:
        status_data = json.load(f)
    
    status_data["status"] = status
    status_data["progress"] = progress
    status_data["message"] = message
    status_data["timestamp"] = str(datetime.utcnow())
    
    with open(status_path, "w") as f:
        json.dump(status_data, f)
```

## Status Endpoint

The status endpoint tracks the progress of image processing:

```python
# api/endpoints/status.py
import os
import json
from fastapi import APIRouter, HTTPException, Path

from core.config import settings
from api.models.responses import StatusResponse

router = APIRouter()

@router.get("/status/{image_id}", response_model=StatusResponse)
async def get_processing_status(
    image_id: str = Path(..., description="ID of the uploaded image")
):
    """
    Check the status of an image processing request.
    
    - **image_id**: ID of the uploaded image
    """
    status_path = os.path.join(settings.UPLOAD_DIR, f"{image_id}_status.json")
    
    try:
        with open(status_path, "r") as f:
            status_data = json.load(f)
        
        # Calculate estimated time remaining if in progress
        estimated_time = None
        if status_data["status"] == "processing" and status_data["progress"] > 0:
            # Simple estimation based on progress percentage
            # In a real application this would be more sophisticated
            estimated_time = int((100 - status_data["progress"]) / 10)
        
        return {
            "image_id": status_data["image_id"],
            "status": status_data["status"],
            "progress": status_data["progress"],
            "message": status_data["message"],
            "estimated_time_remaining_seconds": estimated_time
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Status for image ID {image_id} not found"
        )
```

## Results Endpoint

The results endpoint provides access to completed analyses:

```python
# api/endpoints/results.py
import os
import json
from fastapi import APIRouter, HTTPException, Path, Query
from typing import Optional

from core.config import settings
from api.models.responses import ThreadCountResults
from storage.file_storage import get_results_path

router = APIRouter()

@router.get("/results/{image_id}", response_model=ThreadCountResults)
async def get_analysis_results(
    image_id: str = Path(..., description="ID of the analyzed image"),
    include_visualizations: bool = Query(True, description="Include visualization images"),
    visualization_format: str = Query("base64", description="Format for visualizations: 'base64', 'url', or 'none'")
):
    """
    Retrieve the results of a completed thread count analysis.
    
    - **image_id**: ID of the analyzed image
    - **include_visualizations**: Include visualization images
    - **visualization_format**: Format for visualizations
    """
    results_path = get_results_path(image_id)
    
    if not os.path.exists(results_path):
        # Check if processing is still in progress
        status_path = os.path.join(settings.UPLOAD_DIR, f"{image_id}_status.json")
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                status = json.load(f)
            
            if status["status"] != "completed":
                raise HTTPException(
                    status_code=202,
                    detail=f"Analysis in progress ({status['progress']}%). Check status endpoint."
                )
        
        raise HTTPException(
            status_code=404,
            detail=f"Results for image ID {image_id} not found"
        )
    
    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Handle visualizations based on request parameters
    if not include_visualizations:
        results["visualizations"] = {}
    elif visualization_format == "url":
        # Convert base64 to URLs if needed
        for key, _ in results["visualizations"].items():
            results["visualizations"][key] = f"/static/results/{image_id}_{key}.png"
    elif visualization_format == "none":
        results["visualizations"] = {}
    
    return results
```

## Data Model Classes

The application uses Pydantic models for request and response validation:

```python
# api/models/requests.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple

class RegionOfInterest(BaseModel):
    x: int = Field(..., description="X coordinate of ROI top-left corner")
    y: int = Field(..., description="Y coordinate of ROI top-left corner")
    width: int = Field(..., description="Width of ROI in pixels")
    height: int = Field(..., description="Height of ROI in pixels")

class ThreadHighlighting(BaseModel):
    warp_color: List[int] = Field([255, 0, 0], description="RGB color for warp threads")
    weft_color: List[int] = Field([0, 0, 255], description="RGB color for weft threads")

class PredictionOptions(BaseModel):
    orientation_correction: bool = Field(True, description="Auto-correct fabric orientation")
    roi: Optional[RegionOfInterest] = Field(None, description="Region of interest for analysis")
    return_visualization: bool = Field(True, description="Return annotated image")
    thread_highlighting: Optional[ThreadHighlighting] = Field(None, description="Thread visualization colors")

class PredictionRequest(BaseModel):
    image_id: str = Field(..., description="ID of the uploaded image")
    options: Optional[PredictionOptions] = Field(None, description="Processing options")
```

```python
# api/models/responses.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union

class UploadResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    image_id: str = Field(..., description="Unique ID for the uploaded image")
    filename: str = Field(..., description="Original filename")
    upload_timestamp: str = Field(..., description="Upload timestamp")
    message: str = Field(..., description="Success message")

class PredictionResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    image_id: str = Field(..., description="Image ID")
    message: str = Field(..., description="Status message")

class ThreadCountMetrics(BaseModel):
    warp: int = Field(..., description="Warp thread count")
    weft: int = Field(..., description="Weft thread count")
    total: int = Field(..., description="Total thread count")
    confidence: float = Field(..., description="Confidence score (0-1)")

class ThreadDetailMetrics(BaseModel):
    mean_spacing: float = Field(..., description="Average thread spacing in inches")
    std_deviation: float = Field(..., description="Standard deviation of thread spacing")
    threads_per_inch: int = Field(..., description="Threads per inch")

class AnalysisDetails(BaseModel):
    warp: ThreadDetailMetrics
    weft: ThreadDetailMetrics
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

class QualityAssessment(BaseModel):
    grade: str = Field(..., description="Quality grade (Basic, Standard, Premium, etc.)")
    score: int = Field(..., description="Quality score (0-100)")
    confidence: float = Field(..., description="Confidence in assessment (0-1)")

class ThreadCountResults(BaseModel):
    success: bool = Field(True, description="Operation success status")
    image_id: str = Field(..., description="Image ID")
    thread_count: ThreadCountMetrics
    quality_assessment: Optional[QualityAssessment] = None
    analysis_details: AnalysisDetails
    visualizations: Dict[str, str] = Field(
        {}, 
        description="Base64 encoded visualizations or URLs to visualization images"
    )

class StatusResponse(BaseModel):
    image_id: str = Field(..., description="Image ID")
    status: str = Field(..., description="Processing status")
    progress: int = Field(..., description="Processing progress (0-100)")
    message: str = Field(..., description="Status message")
    estimated_time_remaining_seconds: Optional[int] = Field(
        None, 
        description="Estimated time remaining in seconds"
    )
```

## Image Processing Pipeline

The image preprocessing module prepares the fabric image for analysis:

```python
# processing/preprocessing.py
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),
    dpi: int = 300,
    enhance_contrast: bool = True,
    denoise: bool = True
) -> np.ndarray:
    """
    Preprocess a fabric image for thread detection.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for model input
        dpi: Image DPI (dots per inch)
        enhance_contrast: Whether to enhance image contrast
        denoise: Whether to apply denoising
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image from {image_path}")
    
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize image maintaining aspect ratio
    h, w = gray.shape
    if h > w:
        new_h = target_size[0]
        new_w = int(w * (new_h / h))
    else:
        new_w = target_size[1]
        new_h = int(h * (new_w / w))
    
    resized = cv2.resize(gray, (new_w, new_h))
    
    # Create square image with padding
    square = np.zeros(target_size, dtype=np.uint8)
    offset_x = (target_size[1] - new_w) // 2
    offset_y = (target_size[0] - new_h) // 2
    square[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    # Apply contrast enhancement
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        square = clahe.apply(square)
    
    # Apply denoising
    if denoise:
        square = cv2.fastNlMeansDenoising(square, None, h=10, searchWindowSize=21)
    
    # Normalize
    normalized = square.astype(np.float32) / 255.0
    
    return normalized
```

## Thread Detection Module

This module handles the detection and counting of threads:

```python
# processing/thread_detection.py
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
import time

from core.config import settings
from ml.inference import run_inference
from processing.preprocessing import preprocess_image
from processing.thread_counting import count_threads, separate_warp_weft

def process_image(
    image_path: str,
    dpi: int = 300,
    **options
) -> np.ndarray:
    """
    Process image through preprocessing pipeline.
    
    Args:
        image_path: Path to the image file
        dpi: Image DPI
        options: Additional preprocessing options
    
    Returns:
        Preprocessed image ready for model inference
    """
    # Preprocess the image
    preprocessed = preprocess_image(
        image_path=image_path,
        target_size=settings.MODEL_INPUT_SIZE,
        dpi=dpi,
        **options
    )
    
    return preprocessed

def analyze_thread_mask(
    thread_mask: np.ndarray,
    original_image_path: str,
    dpi: int = 300,
    **options
) -> Dict[str, Any]:
    """
    Analyze thread mask to count threads and generate results.
    
    Args:
        thread_mask: Binary segmentation mask from model
        original_image_path: Path to original image for visualization
        dpi: Image DPI
        options: Additional analysis options
    
    Returns:
        Dictionary containing analysis results
    """
    start_time = time.time()
    
    # Load original image for visualization
    original_image = cv2.imread(original_image_path)
    
    # Separate warp and weft threads
    warp_mask, weft_mask = separate_warp_weft(thread_mask)
    
    # Count threads
    warp_count, warp_details = count_threads(warp_mask, orientation='warp', dpi=dpi)
    weft_count, weft_details = count_threads(weft_mask, orientation='weft', dpi=dpi)
    
    # Calculate total thread count
    total_count = warp_count + weft_count
    
    # Determine quality grade and score
    quality_assessment = assess_quality(total_count, warp_count, weft_count)
    
    # Create visualizations
    visualizations = generate_visualizations(
        original_image, thread_mask, warp_mask, weft_mask,
        thread_highlighting=options.get('thread_highlighting', {})
    )
    
    # Prepare results
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    results = {
        "success": True,
        "image_id": os.path.basename(original_image_path).split('.')[0],
        "thread_count": {
            "warp": warp_count,
            "weft": weft_count,
            "total": total_count,
            "confidence": min(warp_details["confidence"], weft_details["confidence"])
        },
        "quality_assessment": quality_assessment,
        "analysis_details": {
            "warp": {
                "mean_spacing": warp_details["mean_spacing"],
                "std_deviation": warp_details["std_deviation"],
                "threads_per_inch": warp_count
            },
            "weft": {
                "mean_spacing": weft_details["mean_spacing"],
                "std_deviation": weft_details["std_deviation"],
                "threads_per_inch": weft_count
            },
            "processing_time_ms": processing_time_ms
        },
        "visualizations": visualizations
    }
    
    return results
```

## Thread Counting Module

This module implements algorithms to count threads in segmentation masks:

```python
# processing/thread_counting.py
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from scipy.signal import find_peaks
import math

def separate_warp_weft(thread_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate warp (vertical) and weft (horizontal) threads.
    
    Args:
        thread_mask: Binary segmentation mask
    
    Returns:
        Tuple of (warp_mask, weft_mask)
    """
    # Create structuring elements for morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    
    # Convert float mask to uint8 if needed
    if thread_mask.dtype != np.uint8:
        mask = (thread_mask * 255).astype(np.uint8)
    else:
        mask = thread_mask.copy()
    
    # Threshold if not already binary
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Extract horizontal (weft) threads
    weft_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Extract vertical (warp) threads
    warp_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    
    return warp_mask, weft_mask

def count_threads(
    thread_mask: np.ndarray, 
    orientation: str = 'warp',
    dpi: int = 300
) -> Tuple[int, Dict[str, Any]]:
    """
    Count threads in a binary mask using projection profiles.
    
    Args:
        thread_mask: Binary mask with threads
        orientation: Either 'warp' for vertical or 'weft' for horizontal
        dpi: Image DPI (dots per inch)
    
    Returns:
        Tuple of (thread_count, details_dict)
    """
    # Ensure binary mask
    if thread_mask.max() > 1:
        _, binary_mask = cv2.threshold(thread_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = thread_mask.copy()
    
    # Calculate projection profile
    if orientation == 'warp':
        # For vertical threads, sum across rows
        projection = np.sum(binary_mask, axis=0)
    else:
        # For horizontal threads, sum across columns
        projection = np.sum(binary_mask, axis=1)
    
    # Normalize projection
    if projection.max() > 0:
        projection = projection / projection.max()
    
    # Find peaks in projection
    peaks, properties = find_peaks(
        projection,
        height=0.2,  # Minimum height of peak
        distance=5,  # Minimum distance between peaks
        prominence=0.1  # Minimum prominence
    )
    
    # Calculate thread count based on DPI and image size
    img_size = binary_mask.shape[1 if orientation == 'warp' else 0]
    img_size_inches = img_size / dpi
    
    # Get thread count per inch
    thread_count = int(round(len(peaks) / img_size_inches))
    
    # Calculate thread spacing details
    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        mean_spacing_px = np.mean(peak_distances)
        std_spacing_px = np.std(peak_distances)
        
        # Convert to inches
        mean_spacing_inches = mean_spacing_px / dpi
        std_spacing_inches = std_spacing_px / dpi
        
        # Calculate confidence based on spacing consistency
        cv = std_spacing_px / mean_spacing_px if mean_spacing_px > 0 else 1
        confidence = max(0, min(1, 1 - cv))
    else:
        mean_spacing_inches = 0
        std_spacing_inches = 0
        confidence = 0
    
    details = {
        "num_threads_detected": len(peaks),
        "mean_spacing": mean_spacing_inches,
        "std_deviation": std_spacing_inches,
        "confidence": confidence,
        "peak_positions": peaks.tolist()
    }
    
    return thread_count, details
```

## ML Model Integration

The model module handles loading and inference with the U-Net model:

```python
# ml/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional
import logging

from core.config import settings

logger = logging.getLogger(__name__)

# Define global variable for model
_model = None

class UNetModel(nn.Module):
    """U-Net model for thread segmentation."""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetModel, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._encoder_block(in_channels, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.dec1 = self._decoder_block(1024 + 512, 512)
        self.dec2 = self._decoder_block(512 + 256, 256)
        self.dec3 = self._decoder_block(256 + 128, 128)
        self.dec4 = self._decoder_block(128 + 64, 64)
        
        # Final output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Encoder path with skip connections
        enc1_out = self.enc1[:-1](x)  # Skip max pooling
        x = self.enc1[-1](enc1_out)   # Apply max pooling
        
        enc2_out = self.enc2[:-1](x)
        x = self.enc2[-1](enc2_out)
        
        enc3_out = self.enc3[:-1](x)
        x = self.enc3[-1](enc3_out)
        
        enc4_out = self.enc4[:-1](x)
        x = self.enc4[-1](enc4_out)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.dec1(torch.cat([x, enc4_out], dim=1))
        x = self.dec2(torch.cat([x, enc3_out], dim=1))
        x = self.dec3(torch.cat([x, enc2_out], dim=1))
        x = self.dec4(torch.cat([x, enc1_out], dim=1))
        
        # Final output with sigmoid activation
        x = torch.sigmoid(self.output(x))
        
        return x

def initialize_model() -> None:
    """Initialize and load the thread detection model."""
    global _model
    
    # Check if model already loaded
    if _model is not None:
        return
    
    try:
        # Initialize model architecture
        _model = UNetModel(in_channels=1, out_channels=1)
        
        # Check if model file exists
        if os.path.isfile(settings.MODEL_PATH):
            # Load model weights
            _model.load_state_dict(torch.load(
                settings.MODEL_PATH, 
                map_location=torch.device('cpu')
            ))
            _model.eval()  # Set to evaluation mode
            logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model file not found: {settings.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def get_model() -> Optional[nn.Module]:
    """Get the loaded model instance."""
    global _model
    return _model
```

```python
# ml/inference.py
import torch
import numpy as np
from typing import Union, Dict, Any
import cv2

from ml.model import get_model

def run_inference(
    preprocessed_image: np.ndarray,
) -> np.ndarray:
    """
    Run inference with the thread detection model.
    
    Args:
        preprocessed_image: Preprocessed image as numpy array
    
    Returns:
        Binary segmentation mask
    """
    # Get the model
    model = get_model()
    if model is None:
        raise RuntimeError("Model not initialized")
    
    # Convert to tensor
    if len(preprocessed_image.shape) == 2:
        # Add batch and channel dimensions
        input_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0).unsqueeze(0)
    elif len(preprocessed_image.shape) == 3:
        # Assume single channel image with batch dimension
        input_tensor = torch.from_numpy(preprocessed_image).unsqueeze(1)
    else:
        raise ValueError(f"Unexpected input shape: {preprocessed_image.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert output to numpy array
    thread_mask = output[0, 0].cpu().numpy()
    
    # Apply threshold to get binary mask
    binary_mask = (thread_mask > 0.5).astype(np.uint8) * 255
    
    return binary_mask
```

## Visualization Generator

This module creates visual output for thread detection results:

```python
# processing/visualization.py
import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple
import os

from core.config import settings

def generate_visualizations(
    original_image: np.ndarray,
    thread_mask: np.ndarray,
    warp_mask: np.ndarray,
    weft_mask: np.ndarray,
    thread_highlighting: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Generate visualization images for thread detection results.
    
    Args:
        original_image: Original RGB image
        thread_mask: Combined thread segmentation mask
        warp_mask: Warp threads mask
        weft_mask: Weft threads mask
        thread_highlighting: Optional color settings
    
    Returns:
        Dictionary of base64 encoded visualizations
    """
    # Default thread colors
    warp_color = thread_highlighting.get('warp_color', [255, 0, 0])  # Red
    weft_color = thread_highlighting.get('weft_color', [0, 0, 255])  # Blue
    
    # Resize masks to match original image if needed
    h, w = original_image.shape[:2]
    thread_mask_resized = cv2.resize(thread_mask, (w, h))
    warp_mask_resized = cv2.resize(warp_mask, (w, h))
    weft_mask_resized = cv2.resize(weft_mask, (w, h))
    
    # Create visualization images
    original_vis = original_image.copy()
    
    # Create overlay image
    overlay = original_image.copy()
    
    # Add warp threads (vertical) in red
    overlay[warp_mask_resized > 0] = warp_color
    
    # Add weft threads (horizontal) in blue
    overlay[weft_mask_resized > 0] = weft_color
    
    # Create warp-only visualization
    warp_vis = original_image.copy()
    warp_vis[warp_mask_resized > 0] = warp_color
    
    # Create weft-only visualization
    weft_vis = original_image.copy()
    weft_vis[weft_mask_resized > 0] = weft_color
    
    # Encode images to base64
    visualizations = {
        "original": _encode_image_base64(original_vis),
        "thread_detection": _encode_image_base64(overlay),
        "warp_detection": _encode_image_base64(warp_vis),
        "weft_detection": _encode_image_base64(weft_vis)
    }
    
    return visualizations

def _encode_image_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 encoded string."""
    _, buffer = cv2.imencode('.png', image)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
```

## Error Handling

The application uses custom error classes for better error handling:

```python
# core/errors.py
from typing import Dict, Any, Optional

class APIError(Exception):
    """Base API error class."""
    
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(message)

class ImageProcessingError(APIError):
    """Error raised during image processing."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=422,
            error_code="IMAGE_PROCESSING_ERROR",
            message=message,
            details=details
        )

class ModelInferenceError(APIError):
    """Error raised during model inference."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=500,
            error_code="MODEL_INFERENCE_ERROR",
            message=message,
            details=details
        )

class ResourceNotFoundError(APIError):
    """Error raised when a requested resource is not found."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            message=f"{resource_type} with ID '{resource_id}' not found",
            details=details
        )
```

## Docker Configuration

For easy deployment, the application includes Docker configuration:

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/static/results

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=False

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```
# docker-compose.yml
version: '3'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./static:/app/static
    environment:
      - DEBUG=False
      - HOST=0.0.0.0
      - PORT=8000
    restart: unless-stopped
```

## Deployment Considerations

The backend can be deployed in various environments:

### Development Environment

```bash
# Run in development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Environment

For production deployment, consider:

1. **Gunicorn with Uvicorn workers**:
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   ```

2. **Docker Deployment**:
   ```bash
   docker-compose up -d
   ```

3. **Cloud Services**:
   - AWS Elastic Beanstalk
   - Google Cloud Run
   - Azure App Service

## Performance Optimization

The backend implements several performance optimizations:

1. **Model Quantization**: Reducing model size and inference time
2. **Batch Processing**: Processing multiple images in parallel
3. **Caching**: Caching processed results for repeated requests
4. **Background Tasks**: Non-blocking asynchronous processing
5. **Connection Pooling**: Efficient database connections

## Security Considerations

The application implements security best practices:

1. **Input Validation**: All user inputs are validated using Pydantic models
2. **File Upload Limits**: Restrictions on file size and types
3. **CORS Configuration**: Proper cross-origin resource sharing setup
4. **Error Handling**: Secure error messages that don't leak implementation details
5. **Rate Limiting**: Protection against abuse and DoS attacks

---

For comprehensive details, API documentation, and advanced implementation guides, refer to the backend project repository and code documentation.
