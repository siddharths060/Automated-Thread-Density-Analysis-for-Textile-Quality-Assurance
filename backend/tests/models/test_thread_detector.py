"""
Tests for thread_detector.py model.
"""
import os
import pytest
import numpy as np
from PIL import Image
from models.thread_detector import ThreadDetector


class TestThreadDetector:
    """Test cases for ThreadDetector class."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        # Create an instance of the thread detector
        self.detector = ThreadDetector()
        
        # Create a test image
        test_img_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_pattern.jpg')
        
        # Store the path
        self.test_img_path = test_img_path
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Load test image
        img = np.array(Image.open(self.test_img_path))
        
        # Preprocess image
        processed_img = self.detector._preprocess_image(img)
        
        # Verify the preprocessing result
        assert processed_img is not None
        assert isinstance(processed_img, np.ndarray)
        
        # Should have the same dimensions as input, but possibly different channel count
        assert processed_img.shape[:2] == img.shape[:2]
    
    def test_detect_threads(self):
        """Test thread detection."""
        # Load test image
        img = np.array(Image.open(self.test_img_path))
        
        # Detect threads
        result = self.detector.detect_threads(img)
        
        # Verify the result
        assert 'warp_lines' in result
        assert 'weft_lines' in result
        assert len(result['warp_lines']) > 0
        assert len(result['weft_lines']) > 0
    
    def test_detect_with_invalid_input(self):
        """Test behavior with invalid input."""
        # Test with None input
        with pytest.raises(ValueError):
            self.detector.detect_threads(None)
        
        # Test with empty array
        with pytest.raises(ValueError):
            self.detector.detect_threads(np.array([]))
        
        # Test with wrong dimensions
        with pytest.raises(ValueError):
            self.detector.detect_threads(np.zeros((10,)))  # 1D array
    
    def test_confidence_calculation(self):
        """Test confidence calculation for thread detection."""
        # Create a sample line detection result
        lines = list(range(10, 100, 10))  # Evenly spaced lines
        
        # Calculate confidence
        confidence = self.detector._calculate_confidence(lines, expected_range=(5, 15))
        
        # Verify the confidence is within range [0,1]
        assert 0 <= confidence <= 1
        
        # Confidence should be high for evenly spaced lines
        assert confidence > 0.7
