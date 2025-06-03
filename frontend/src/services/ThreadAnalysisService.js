import axios from 'axios';

// Create API instance with default configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Create an axios instance with custom configuration
const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 60000, // 60 second timeout
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  }
});

// Add request interceptor for global error handling
apiClient.interceptors.request.use(
  (config) => {
    // You could add authentication tokens here
    return config;
  },
  (error) => {
    console.error('API request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for global error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle common errors
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout');
      error.customMessage = 'The server took too long to respond. Please try again.';
    } else if (!error.response) {
      console.error('Network error', error);
      error.customMessage = 'Network error. Please check your connection and try again.';
    } else if (error.response.status === 413) {
      error.customMessage = 'The image file is too large. Please upload a smaller file (max 10MB).';
    }
    
    return Promise.reject(error);
  }
);

class ThreadAnalysisService {
  /**
   * Upload an image file to the server
   * @param {File} file - The image file to upload
   * @param {Function} onUploadProgress - Callback for upload progress updates
   * @returns {Promise} - Promise with the upload response
   */
  async uploadImage(file, onUploadProgress = null) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await apiClient.post(`/analysis/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: onUploadProgress ? (progressEvent) => {
          onUploadProgress(progressEvent);
        } : undefined
      });
      
      return response.data;
    } catch (error) {
      console.error('Error uploading image:', error);
      
      // If we have a custom message from the interceptor, use it
      if (error.customMessage) {
        const customError = new Error(error.customMessage);
        customError.originalError = error;
        throw customError;
      }
      throw error;
    }
  }
  
  /**
   * Analyze an uploaded image for thread density
   * @param {string} imagePath - Path to the uploaded image
   * @param {Function} onAnalysisProgress - Optional callback for analysis progress
   * @returns {Promise} - Promise with the analysis results
   */
  async analyzeImage(imagePath, onAnalysisProgress = null) {
    try {
      // Set timeout for analysis which may take longer
      const response = await apiClient.post(`/analysis/predict`, {
        image_path: imagePath
      }, {
        timeout: 120000, // 2 minute timeout for analysis
      });
      
      return response.data;
    } catch (error) {
      console.error('Error analyzing image:', error);
      
      // If we have a custom message from the interceptor, use it
      if (error.customMessage) {
        const customError = new Error(error.customMessage);
        customError.originalError = error;
        throw customError;
      }
      throw error;
    }
  }
}

export default new ThreadAnalysisService();
