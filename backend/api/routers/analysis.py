"""
Router for thread density analysis endpoints.

This module provides API endpoints for thread density analysis operations,
including image upload and thread detection functionality.
"""
import os
import time
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, status
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import uuid
import shutil
import imghdr

from api.models.analysis import ThreadAnalysisRequest, ThreadAnalysisResponse, ErrorDetail
from services.thread_analysis import ThreadAnalysisService
from utils.logging_config import get_logger

# Initialize logger
logger = get_logger("api.routers.analysis")

# Create router with standard prefix
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Initialize the thread analysis service
thread_analysis_service = ThreadAnalysisService()

# Define allowed image formats
ALLOWED_IMAGE_FORMATS = {"jpeg", "jpg", "png", "bmp", "tiff"}


@router.post("/upload", response_model=ThreadAnalysisRequest, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for fabric thread analysis.
    
    Args:
        file: The image file uploaded by the client.
        
    Returns:
        ThreadAnalysisRequest: Object containing the path to the uploaded image.
        
    Raises:
        HTTPException: If the upload fails for any reason (no file, invalid format, etc.).
    """
    logger.info(f"Received upload request for file: {file.filename}")
    
    # Validate file presence
    if not file.filename:
        logger.warning("Upload attempted with no file")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded"
        )
    
    # Get file content for validation
    file_content = await file.read()
    
    # Return file pointer to start
    await file.seek(0)
    
    # Validate file size (max 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    if len(file_content) > max_size:
        logger.warning(f"Upload rejected: file size {len(file_content)} exceeds limit of {max_size}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of 25MB"
        )
    
    # Validate content type using imghdr instead of just extension
    image_format = imghdr.what(None, file_content)
    
    if not image_format or image_format not in ALLOWED_IMAGE_FORMATS:
        logger.warning(f"Upload rejected: invalid format {image_format}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Invalid file format. Only JPEG, PNG, BMP, and TIFF images are allowed."
        )
    
    # Use original extension if valid, otherwise use detected format
    file_extension = os.path.splitext(file.filename)[1].lower()
    if not file_extension or file_extension[1:] not in ALLOWED_IMAGE_FORMATS:
        file_extension = f".{image_format}"
    
    # Create a unique filename with UUID to prevent collisions
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    upload_dir = "uploads"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Ensure upload directory exists with proper permissions
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File successfully saved to {file_path}")
    except IOError as e:
        logger.error(f"IO error saving file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file due to IO error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error saving file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while saving file: {str(e)}"
        )
    
    # Return the image path for the next step
    return ThreadAnalysisRequest(image_path=file_path)


@router.post("/predict", response_model=ThreadAnalysisResponse)
async def analyze_threads(request: ThreadAnalysisRequest):
    """
    Analyze thread density in the uploaded fabric image.
    
    This endpoint processes the uploaded image to detect and count warp and weft threads,
    calculates thread density metrics, and returns the results along with an annotated image.
    
    Args:
        request: ThreadAnalysisRequest containing the path to the uploaded image.
    
    Returns:
        ThreadAnalysisResponse: Analysis results including thread counts, density metrics,
                              and path to the annotated image.
    
    Raises:
        HTTPException: If the image cannot be found or processed.
    """
    start_time = time.time()
    req_id = str(uuid.uuid4())[:8]  # Generate short request ID for tracking
    logger.info(f"[{req_id}] Analysis request received for image: {request.image_path}")
    
    try:
        # Validate image exists
        if not os.path.exists(request.image_path):
            logger.warning(f"[{req_id}] Image not found: {request.image_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Image not found or has been deleted"
            )
        
        # Validate image is readable
        try:
            with open(request.image_path, "rb") as f:
                if imghdr.what(None, f.read(1024)) not in ALLOWED_IMAGE_FORMATS:
                    logger.warning(f"[{req_id}] Invalid image format: {request.image_path}")
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="Invalid image format or corrupted file"
                    )
        except IOError as e:
            logger.error(f"[{req_id}] Error reading image: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error reading image file: {str(e)}"
            )
        
        # Process the image with detailed error handling
        logger.info(f"[{req_id}] Starting thread analysis")
        try:
            result = thread_analysis_service.analyze_image(request.image_path)
            logger.info(f"[{req_id}] Thread analysis completed successfully")
        except ValueError as e:
            logger.error(f"[{req_id}] Value error in thread analysis: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Image analysis failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"[{req_id}] Unexpected error in thread analysis: {str(e)}", exc_info=True)
            error_detail = {
                "type": type(e).__name__,
                "message": str(e),
                "trace_id": req_id
            }
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorDetail(**error_detail).dict()
            )
        
        processing_time = time.time() - start_time
        logger.info(f"[{req_id}] Analysis completed in {processing_time:.2f} seconds")
        
        # Return the analysis results
        return ThreadAnalysisResponse(
            success=True,
            message="Thread analysis completed successfully",
            original_image_path=request.image_path,
            annotated_image_path=result["annotated_image_path"],
            results=result["thread_count"],
            processing_time=processing_time
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    
    except Exception as e:
        # Catch any other unexpected exceptions
        processing_time = time.time() - start_time
        error_trace = traceback.format_exc()
        logger.error(f"[{req_id}] Unhandled exception: {str(e)}\n{error_trace}")
        
        # Return a failure response with error details
        return ThreadAnalysisResponse(
            success=False,
            message="Thread analysis failed due to an unexpected error",
            error=f"Error processing image: {str(e)}",
            processing_time=processing_time
        )
