"""
Automated Thread Density Analysis - Main Application

This module initializes the FastAPI application and includes all routers.
"""
import os
import logging
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.routers import analysis
from utils.security_utils import RateLimiter, SecurityHeadersMiddleware
from utils.logging_config import get_logger

# Initialize logger
logger = get_logger("main")

# Create FastAPI app
app = FastAPI(
    title="Thread Density Analysis API",
    description="API for analyzing thread density in textile images",
    version="1.0.0",
    docs_url="/api/docs",  # Change Swagger UI path for security
    redoc_url="/api/redoc",  # Change ReDoc path
    openapi_url="/api/openapi.json"  # Change OpenAPI path
)

# Configure CORS with more restrictive settings
allowed_origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:5000",  # Production frontend
    "https://thread-analysis.example.com"  # Replace with actual production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimiter, max_requests=30, window_seconds=60)  # Adjust as needed

# Global exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with sanitized output."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid input data. Please check your request format."},
    )

# Generic exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors with sanitized output."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# Include routers
app.include_router(analysis.router)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
# Set proper permissions for security
os.chmod("uploads", 0o755)
os.chmod("results", 0o755)

# Mount static files for uploaded images and results
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)