"""
Analysis data models for the thread density analysis API.

This module defines the data models used for request/response in the thread analysis API.
These models are used for validation and serialization with Pydantic.
"""
from datetime import datetime
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union


class ThreadAnalysisRequest(BaseModel):
    """Request model for thread analysis."""
    image_path: str = Field(..., description="Path to the uploaded image")
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate image path is not empty."""
        if not v.strip():
            raise ValueError("Image path cannot be empty")
        return v


class ThreadCountResult(BaseModel):
    """Thread count results with detailed metrics."""
    warp_count: int = Field(..., description="Number of warp (vertical) threads")
    weft_count: int = Field(..., description="Number of weft (horizontal) threads")
    total_count: int = Field(..., description="Total thread count")
    density: float = Field(..., description="Thread density (threads per inch/cm)")
    unit: str = Field("cm", description="Unit of measurement (cm or inch)")
    
    @validator('warp_count', 'weft_count', 'total_count')
    def validate_counts(cls, v):
        """Validate thread counts are non-negative."""
        if v < 0:
            raise ValueError("Thread count cannot be negative")
        return v
    
    @validator('density')
    def validate_density(cls, v):
        """Validate density is positive."""
        if v <= 0:
            raise ValueError("Thread density must be positive")
        return v
    
    @validator('unit')
    def validate_unit(cls, v):
        """Validate unit is either cm or inch."""
        if v not in ["cm", "inch"]:
            raise ValueError("Unit must be either 'cm' or 'inch'")
        return v


class ErrorDetail(BaseModel):
    """Detailed error information."""
    type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    trace_id: str = Field(..., description="Unique trace ID for error tracking")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of error")


class ThreadAnalysisResponse(BaseModel):
    """Response model for thread analysis with detailed error handling."""
    success: bool = Field(..., description="Whether the analysis was successful")
    message: str = Field(..., description="Status message")
    original_image_path: Optional[str] = Field(None, description="Path to the original image")
    annotated_image_path: Optional[str] = Field(None, description="Path to the annotated image")
    results: Optional[ThreadCountResult] = Field(None, description="Analysis results")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Error message or details if analysis failed")
