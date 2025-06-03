"""
Security utilities for the thread density analysis application.

This module provides security-related functions and configurations to enhance
the application's security posture, including rate limiting, input validation,
and security headers.
"""
import time
import secrets
import re
from typing import Dict, Callable, Any, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Rate limiting middleware to prevent abuse."""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_records: Dict[str, list] = {}
    
    def _generate_client_id(self, request: Request) -> str:
        """
        Generate a client identifier based on IP and user agent.
        
        Args:
            request: The incoming request
            
        Returns:
            A unique identifier for the client
        """
        ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        return f"{ip}:{user_agent}"
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """
        Check if a client has exceeded rate limits.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if rate limited, False otherwise
        """
        now = time.time()
        
        # Initialize client record if not exists
        if client_id not in self.request_records:
            self.request_records[client_id] = []
        
        # Filter out old requests
        self.request_records[client_id] = [
            timestamp for timestamp in self.request_records[client_id]
            if now - timestamp <= self.window_seconds
        ]
        
        # Check if exceeded limit
        if len(self.request_records[client_id]) >= self.max_requests:
            return True
        
        # Add current request timestamp
        self.request_records[client_id].append(now)
        return False
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and apply rate limiting.
        
        Args:
            request: The incoming request
            call_next: The next middleware or endpoint handler
            
        Returns:
            The response from the next handler
            
        Raises:
            HTTPException: If client is rate limited
        """
        # Skip rate limiting for static files
        if request.url.path.startswith("/static") or request.url.path.startswith("/uploads"):
            return await call_next(request)
        
        client_id = self._generate_client_id(request)
        
        if self._is_rate_limited(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to the response.
        
        Args:
            request: The incoming request
            call_next: The next middleware or endpoint handler
            
        Returns:
            The response with added security headers
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
        
        return response


def sanitize_input(input_string: Optional[str]) -> Optional[str]:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_string: The input string to sanitize
        
    Returns:
        Sanitized string
    """
    if input_string is None:
        return None
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>&;]', '', input_string)
    
    # Limit the length
    return sanitized[:1000] if sanitized else sanitized


def generate_secure_token() -> str:
    """
    Generate a cryptographically secure token.
    
    Returns:
        Secure token string
    """
    return secrets.token_urlsafe(32)


def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """
    Validate file extension against a list of allowed extensions.
    
    Args:
        filename: File name to validate
        allowed_extensions: List of allowed file extensions
        
    Returns:
        True if valid, False otherwise
    """
    if not filename or '.' not in filename:
        return False
    
    ext = filename.split('.')[-1].lower()
    return ext in allowed_extensions
