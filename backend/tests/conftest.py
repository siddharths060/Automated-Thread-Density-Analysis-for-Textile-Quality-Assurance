"""Configuration file for pytest."""
import os
import sys
import pytest

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test fixtures that can be reused across tests
@pytest.fixture
def test_image_path():
    """Return path to a test image file."""
    return os.path.join(os.path.dirname(__file__), 'data', 'test_fabric.jpg')

@pytest.fixture
def test_invalid_image_path():
    """Return path to an invalid test file."""
    return os.path.join(os.path.dirname(__file__), 'data', 'invalid_file.txt')

# Ensure the data directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)

# Create an empty test client fixture for FastAPI
@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    from fastapi.testclient import TestClient
    from main import app
    
    return TestClient(app)
