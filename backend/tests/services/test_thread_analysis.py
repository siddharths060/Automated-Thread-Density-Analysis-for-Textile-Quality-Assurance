"""
Tests for thread_analysis.py service.
"""
import os
import pytest
from services.thread_analysis import ThreadAnalysisService
from PIL import Image
import numpy as np


class TestThreadAnalysisService:
    """Test cases for ThreadAnalysisService."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        # Create an instance of the service
        self.service = ThreadAnalysisService()
        
        # Create a test image with a known pattern if it doesn't exist
        test_pattern_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'test_pattern.jpg')
        if not os.path.exists(test_pattern_path):
            # Create a simple pattern with horizontal and vertical lines
            img_size = 200
            img_array = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            # Add horizontal lines (weft threads)
            line_spacing = 20
            line_thickness = 3
            for i in range(0, img_size, line_spacing):
                img_array[i:i+line_thickness, :, :] = [0, 0, 0]  # Black lines
            
            # Add vertical lines (warp threads)
            for i in range(0, img_size, line_spacing):
                img_array[:, i:i+line_thickness, :] = [0, 0, 0]  # Black lines
            
            # Save the pattern
            img = Image.fromarray(img_array)
            img.save(test_pattern_path)
            
        # Store the path for tests
        self.test_pattern_path = test_pattern_path
        
        # Create an invalid text file
        invalid_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'invalid_file.txt')
        with open(invalid_path, 'w') as f:
            f.write("This is not an image file")
            
        self.invalid_file_path = invalid_path
    
    def test_analyze_image_with_valid_image(self):
        """Test image analysis with a valid image."""
        # Analyze the test pattern
        result = self.service.analyze_image(self.test_pattern_path)
        
        # Verify the result structure
        assert 'thread_count' in result
        assert 'annotated_image_path' in result
        
        # Verify thread count data
        thread_count = result['thread_count']
        assert thread_count['warp_count'] > 0
        assert thread_count['weft_count'] > 0
        assert thread_count['total_count'] == thread_count['warp_count'] + thread_count['weft_count']
        assert thread_count['density'] > 0
        assert thread_count['unit'] in ['cm', 'inch']
        
        # Verify annotated image was created
        assert os.path.exists(result['annotated_image_path'])
    
    def test_analyze_image_with_real_photo(self, test_image_path):
        """Test image analysis with a real photo."""
        # Skip this test if the file doesn't exist
        if not os.path.exists(test_image_path):
            pytest.skip(f"Test image not found at {test_image_path}")
        
        # Analyze the real photo
        result = self.service.analyze_image(test_image_path)
        
        # Verify the result structure
        assert 'thread_count' in result
        assert 'annotated_image_path' in result
        
        # Verify annotated image was created
        assert os.path.exists(result['annotated_image_path'])
    
    def test_analyze_image_with_invalid_file(self):
        """Test behavior with invalid image file."""
        # Should raise ValueError for non-image file
        with pytest.raises(ValueError, match=r".*Invalid image file.*"):
            self.service.analyze_image(self.invalid_file_path)
    
    def test_analyze_image_with_nonexistent_file(self):
        """Test behavior with nonexistent file."""
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            self.service.analyze_image("/path/to/nonexistent/image.jpg")
    
    def test_thread_detection(self):
        """Test the thread detection function."""
        # Load the test pattern
        img = np.array(Image.open(self.test_pattern_path))
        
        # Detect threads
        warp_lines, weft_lines = self.service._detect_threads(img)
        
        # Verify lines were detected
        assert len(warp_lines) > 0
        assert len(weft_lines) > 0
    
    def test_calculate_thread_metrics(self):
        """Test the thread metrics calculation."""
        # Create fake line coordinates
        warp_lines = [10, 30, 50, 70, 90]  # 5 lines
        weft_lines = [15, 35, 55, 75]  # 4 lines
        image_width = 100
        image_height = 100
        
        # Calculate metrics
        metrics = self.service._calculate_thread_metrics(
            warp_lines, weft_lines, image_width, image_height
        )
        
        # Verify the metrics
        assert metrics['warp_count'] == 5
        assert metrics['weft_count'] == 4
        assert metrics['total_count'] == 9
        assert metrics['density'] > 0
        assert metrics['unit'] in ['cm', 'inch']
