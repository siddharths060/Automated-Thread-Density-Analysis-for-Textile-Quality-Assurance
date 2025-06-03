import React, { useState, useCallback, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  CircularProgress,
  Alert,
  Snackbar,
  LinearProgress,
  Fade,
  Backdrop
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import InfoIcon from '@mui/icons-material/Info';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

import ThreadAnalysisService from '../services/ThreadAnalysisService';

// Maximum file size in bytes (10MB)
const MAX_FILE_SIZE = 10 * 1024 * 1024;

// Accepted file types
const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/bmp': ['.bmp'],
  'image/tiff': ['.tiff', '.tif']
};

const UploadImage = ({ onFileUpload, onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [successMessage, setSuccessMessage] = useState('');
  
  // Clear any success message after 5 seconds
  useEffect(() => {
    let timer;
    if (successMessage) {
      timer = setTimeout(() => {
        setSuccessMessage('');
      }, 5000);
    }
    return () => clearTimeout(timer);
  }, [successMessage]);
  
  const validateFile = (file) => {
    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      return 'File size exceeds 10MB limit. Please upload a smaller image.';
    }
    
    // Check file type
    if (!file.type.match('image.*')) {
      return 'Please upload an image file (JPEG, PNG, BMP, TIFF).';
    }
    
    return null;
  };
  
  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    const validationError = validateFile(file);
    
    if (validationError) {
      setError(validationError);
      return;
    }
    
    // Display preview
    const preview = URL.createObjectURL(file);
    setPreviewImage(preview);
    setError(null);
    onFileUpload(file);
    
    // Upload to server
    setUploading(true);
    setUploadProgress(0);
    
    try {
      const response = await ThreadAnalysisService.uploadImage(file, (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
      });
      
      setSuccessMessage('Image uploaded successfully!');
      onUploadSuccess(response);
    } catch (err) {
      console.error('Upload error:', err);
      setError(
        err.response?.data?.detail || 
        'Error uploading image. Please try again.'
      );
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [onFileUpload, onUploadSuccess]);
  
  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_FILE_SIZE,
    multiple: false
  });
  
  // Handle clearImage
  const clearImage = () => {
    setPreviewImage(null);
    setError(null);
    if (previewImage) {
      URL.revokeObjectURL(previewImage); // Clean up memory
    }
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <CloudUploadIcon sx={{ mr: 1 }} />
        Upload a Fabric Image
      </Typography>
      
      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<ErrorOutlineIcon fontSize="inherit" />}
        >
          {error}
        </Alert>
      )}
      
      {successMessage && (
        <Snackbar
          open={!!successMessage}
          autoHideDuration={5000}
          onClose={() => setSuccessMessage('')}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
          <Alert severity="success" sx={{ width: '100%' }}>
            {successMessage}
          </Alert>
        </Snackbar>
      )}
      
      <Paper
        {...getRootProps()}
        sx={{
          p: 3,
          textAlign: 'center',
          backgroundColor: isDragActive 
            ? 'rgba(25, 118, 210, 0.08)' 
            : isDragReject
              ? 'rgba(211, 47, 47, 0.08)'
              : 'inherit',
          border: '2px dashed',
          borderColor: isDragActive 
            ? 'primary.main' 
            : isDragReject
              ? 'error.main'
              : error 
                ? 'error.main' 
                : 'grey.400',
          borderRadius: 2,
          cursor: uploading ? 'default' : 'pointer',
          mb: 3,
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: uploading ? 'inherit' : 'rgba(0, 0, 0, 0.04)'
          }
        }}
      >
        <input {...getInputProps()} disabled={uploading} />
        
        {uploading ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <CircularProgress size={40} sx={{ mb: 2 }} />
            <Typography>Uploading image...</Typography>
            <Box sx={{ width: '100%', mt: 2 }}>
              <LinearProgress variant="determinate" value={uploadProgress} />
              <Typography variant="caption" sx={{ mt: 0.5 }}>
                {uploadProgress}% complete
              </Typography>
            </Box>
          </Box>
        ) : (
          <Box>
            <Fade in={true}>
              <CloudUploadIcon sx={{ 
                fontSize: 48, 
                color: isDragReject || error ? 'error.main' : 'primary.main', 
                mb: 1 
              }} />
            </Fade>
            <Typography variant="h6">
              {isDragActive
                ? isDragReject 
                  ? 'File type or size not accepted'
                  : 'Drop the image here'
                : 'Drag & drop an image here, or click to select one'
              }
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
              Supports JPEG, PNG, BMP, TIFF (max 10MB)
            </Typography>
            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <InfoIcon sx={{ fontSize: 16, mr: 0.5, color: 'info.main' }} />
              <Typography variant="caption" color="info.main">
                High-quality images yield the most accurate thread counts
              </Typography>
            </Box>
          </Box>
        )}
      </Paper>
      
      {previewImage && !uploading && (
        <Box sx={{ mt: 2, textAlign: 'center', position: 'relative' }}>
          <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <InfoIcon sx={{ mr: 1, fontSize: 18 }} />
            Preview:
          </Typography>
          <Paper elevation={3} sx={{ p: 2, position: 'relative' }}>
            <Box
              component="img"
              src={previewImage}
              alt="Preview"
              sx={{
                maxWidth: '100%',
                maxHeight: '300px',
                objectFit: 'contain',
                borderRadius: 1,
                transition: 'all 0.3s ease-in-out',
              }}
            />
            <Button 
              variant="outlined" 
              color="secondary" 
              size="small" 
              onClick={(e) => {
                e.stopPropagation();
                clearImage();
              }}
              sx={{ 
                position: 'absolute', 
                top: 8, 
                right: 8,
                opacity: 0.7,
                '&:hover': {
                  opacity: 1
                }
              }}
            >
              Change Image
            </Button>
          </Paper>
          
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <Button 
              variant="contained" 
              color="primary" 
              sx={{ mt: 1 }}
              onClick={(e) => {
                e.stopPropagation();
                // Re-trigger the file upload if needed
                if (previewImage && !uploading) {
                  // Implementation depends on specific requirements
                }
              }}
            >
              Confirm Selection
            </Button>
          </Box>
        </Box>
      )}
      
      {/* Backdrop for showing loading state */}
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={uploading}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <CircularProgress color="inherit" />
          <Typography sx={{ mt: 2 }}>Processing your image...</Typography>
        </Box>
      </Backdrop>
    </Box>
  );
};

export default UploadImage;
