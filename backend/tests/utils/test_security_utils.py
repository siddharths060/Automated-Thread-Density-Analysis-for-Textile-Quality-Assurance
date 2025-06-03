"""
Tests for security utilities.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException, status
from fastapi.responses import Response
from fastapi.requests import Request

from utils.security_utils import (
    RateLimiter, SecurityHeadersMiddleware, 
    sanitize_input, validate_file_extension, generate_secure_token
)


class TestRateLimiter:
    """Test cases for RateLimiter middleware."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter instance for testing."""
        return RateLimiter(max_requests=3, window_seconds=60)
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "test-agent"}
        request.url = MagicMock()
        request.url.path = "/test"
        return request
    
    @pytest.fixture
    def mock_call_next(self):
        """Create a mock for the call_next function."""
        async def mock_call_next_func(request):
            return Response(content="Test response")
        return AsyncMock(side_effect=mock_call_next_func)
    
    async def test_allow_requests_under_limit(self, rate_limiter, mock_request, mock_call_next):
        """Test that requests under the limit are allowed."""
        # Make requests up to the limit
        for _ in range(3):
            response = await rate_limiter(mock_request, mock_call_next)
            assert isinstance(response, Response)
        
        # The call_next should have been called 3 times
        assert mock_call_next.call_count == 3
    
    async def test_block_requests_over_limit(self, rate_limiter, mock_request, mock_call_next):
        """Test that requests over the limit are blocked."""
        # Make requests up to the limit
        for _ in range(3):
            await rate_limiter(mock_request, mock_call_next)
        
        # The fourth request should be rate limited
        with pytest.raises(HTTPException) as excinfo:
            await rate_limiter(mock_request, mock_call_next)
            
        assert excinfo.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    
    async def test_skip_static_files(self, rate_limiter, mock_request, mock_call_next):
        """Test that static file requests are not rate limited."""
        # Set the path to a static file
        mock_request.url.path = "/uploads/image.jpg"
        
        # Make more requests than the limit
        for _ in range(5):
            response = await rate_limiter(mock_request, mock_call_next)
            assert isinstance(response, Response)
        
        # No rate limiting should have occurred
        assert mock_call_next.call_count == 5


class TestSecurityHeadersMiddleware:
    """Test cases for SecurityHeadersMiddleware."""
    
    @pytest.fixture
    def security_middleware(self):
        """Create a security headers middleware instance."""
        return SecurityHeadersMiddleware(None)
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        return MagicMock()
    
    @pytest.fixture
    def mock_call_next(self):
        """Create a mock for the call_next function."""
        response = Response(content="Test response")
        return AsyncMock(return_value=response)
    
    async def test_adds_security_headers(self, security_middleware, mock_request, mock_call_next):
        """Test that security headers are added to the response."""
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        # Check each security header
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers


class TestSecurityUtilityFunctions:
    """Test cases for security utility functions."""
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        # Test with potentially harmful input
        harmful_input = "<script>alert('XSS')</script>"
        sanitized = sanitize_input(harmful_input)
        assert "<script>" not in sanitized
        assert "alert" in sanitized  # The text itself is ok
        
        # Test with valid input
        valid_input = "Regular text input"
        assert sanitize_input(valid_input) == valid_input
        
        # Test with None
        assert sanitize_input(None) is None
    
    def test_generate_secure_token(self):
        """Test secure token generation."""
        token1 = generate_secure_token()
        token2 = generate_secure_token()
        
        # Tokens should be strings
        assert isinstance(token1, str)
        assert isinstance(token2, str)
        
        # Tokens should be different each time
        assert token1 != token2
        
        # Tokens should have sufficient length (at least 32 bytes)
        assert len(token1) >= 32
    
    def test_validate_file_extension(self):
        """Test file extension validation."""
        allowed = ["jpg", "jpeg", "png"]
        
        # Test valid extensions
        assert validate_file_extension("image.jpg", allowed) is True
        assert validate_file_extension("photo.png", allowed) is True
        assert validate_file_extension("FILE.JPEG", allowed) is True  # Should be case-insensitive
        
        # Test invalid extensions
        assert validate_file_extension("script.js", allowed) is False
        assert validate_file_extension("document.pdf", allowed) is False
        
        # Test edge cases
        assert validate_file_extension("", allowed) is False
        assert validate_file_extension("noextension", allowed) is False
