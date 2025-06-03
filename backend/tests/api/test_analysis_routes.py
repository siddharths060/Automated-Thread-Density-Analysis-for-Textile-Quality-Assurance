"""
Tests for API endpoints in analysis.py router.
"""
import os
import pytest
from fastapi.testclient import TestClient
from fastapi import status
import io
from PIL import Image
import numpy as np


class TestAnalysisAPI:
    """Test cases for thread analysis API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup(self, test_client):
        """Set up the test environment."""
        self.client = test_client
        
        # Create a test image
        self.test_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_fabric.jpg')
        
        # Create a simple test image if it doesn't exist
        if not os.path.exists(self.test_image_path):
            img_size = 200
            img_array = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            # Add some lines to simulate threads
            for i in range(0, img_size, 20):
                img_array[i:i+2, :, :] = [0, 0, 0]  # Horizontal lines
                img_array[:, i:i+2, :] = [0, 0, 0]  # Vertical lines
            
            # Save the test image
            Image.fromarray(img_array).save(self.test_image_path)
    
    def test_upload_image_endpoint(self):
        """Test the image upload endpoint."""
        # Open the test image
        with open(self.test_image_path, "rb") as img_file:
            # Create form data with the image
            files = {"file": ("test_fabric.jpg", img_file, "image/jpeg")}
            
            # Make the request
            response = self.client.post("/api/analysis/upload", files=files)
            
            # Check response
            assert response.status_code == status.HTTP_201_CREATED
            assert "image_path" in response.json()
            assert response.json()["image_path"].endswith(".jpg")
    
    def test_upload_invalid_file_type(self):
        """Test uploading an invalid file type."""
        # Create a text file
        text_content = io.BytesIO(b"This is not an image")
        
        # Create form data with the text file
        files = {"file": ("test.txt", text_content, "text/plain")}
        
        # Make the request
        response = self.client.post("/api/analysis/upload", files=files)
        
        # Check response - should be an unsupported media type error
        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    
    def test_upload_oversized_file(self):
        """Test uploading a file that exceeds size limit."""
        # Create a large binary file (26MB)
        large_content = io.BytesIO(b"0" * (26 * 1024 * 1024))
        
        # Create form data with the large file
        files = {"file": ("large.jpg", large_content, "image/jpeg")}
        
        # Make the request
        response = self.client.post("/api/analysis/upload", files=files)
        
        # Check response - should be a request entity too large error
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    def test_detect_threads_endpoint(self):
        """Test the thread detection endpoint with a valid image."""
        # First upload an image
        with open(self.test_image_path, "rb") as img_file:
            files = {"file": ("test_fabric.jpg", img_file, "image/jpeg")}
            upload_response = self.client.post("/api/analysis/upload", files=files)
            
        # Get the image path from the upload response
        image_path = upload_response.json()["image_path"]
        
        # Request thread detection
        detect_response = self.client.post(
            "/api/analysis/detect",
            json={"image_path": image_path}
        )
        
        # Check response
        assert detect_response.status_code == status.HTTP_200_OK
        
        # Verify response structure
        result = detect_response.json()
        assert result["success"] is True
        assert "original_image_path" in result
        assert "annotated_image_path" in result
        assert "results" in result
        assert "processing_time" in result
        
        # Verify thread count metrics
        metrics = result["results"]
        assert "warp_count" in metrics
        assert "weft_count" in metrics
        assert "total_count" in metrics
        assert "density" in metrics
        assert "unit" in metrics
    
    def test_detect_with_invalid_path(self):
        """Test thread detection with an invalid image path."""
        # Request thread detection with nonexistent path
        detect_response = self.client.post(
            "/api/analysis/detect",
            json={"image_path": "nonexistent/path.jpg"}
        )
        
        # Check response - should be a not found error
        assert detect_response.status_code == status.HTTP_404_NOT_FOUND
