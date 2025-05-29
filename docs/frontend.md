# 🖥️ Frontend Implementation

This document details the React.js frontend implementation for the Automated Thread Density Analysis system, covering components, state management, API integration, and the user interface.

## Architecture Overview

The frontend is built using:
- **React.js 18+**: Core UI library
- **Material UI**: Component library for consistent design
- **React Router**: Navigation management
- **Axios**: API requests
- **Chart.js**: Data visualization
- **React Query**: Server state management
- **Zustand**: Client state management

![Frontend Architecture](assets/frontend-architecture.png)

## Component Structure

The frontend follows a modular component structure:

```
src/
├── components/
│   ├── common/
│   │   ├── Header.jsx
│   │   ├── Footer.jsx
│   │   ├── LoadingSpinner.jsx
│   │   ├── ErrorBoundary.jsx
│   │   └── ...
│   ├── upload/
│   │   ├── ImageUploader.jsx
│   │   ├── DropZone.jsx
│   │   ├── ImagePreview.jsx
│   │   └── UploadProgress.jsx
│   ├── analysis/
│   │   ├── ThreadCounter.jsx
│   │   ├── ResultsCard.jsx
│   │   ├── ThreadVisualizer.jsx
│   │   └── ThreadChart.jsx
│   └── settings/
│       ├── AnalysisSettings.jsx
│       ├── CalibrationTool.jsx
│       └── ...
├── pages/
│   ├── HomePage.jsx
│   ├── UploadPage.jsx
│   ├── ResultsPage.jsx
│   ├── HistoryPage.jsx
│   └── SettingsPage.jsx
├── services/
│   ├── api.js
│   ├── threadAnalysis.js
│   └── imageProcessing.js
├── hooks/
│   ├── useImageUpload.js
│   ├── useThreadAnalysis.js
│   └── useSettings.js
├── store/
│   ├── settingsStore.js
│   └── analysisStore.js
└── App.jsx
```

## Key Components

### ImageUploader Component

The ImageUploader component handles user image uploads with drag-and-drop functionality:

```jsx
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, CircularProgress, Paper } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImagePreview from './ImagePreview';
import { uploadImage } from '../../services/api';

const ImageUploader = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewImage, setPreviewImage] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    
    // Preview image
    setPreviewImage(URL.createObjectURL(file));
    
    // Reset states
    setUploading(true);
    setUploadProgress(0);
    setError(null);
    
    try {
      // Upload with progress tracking
      const imageId = await uploadImage(file, (progress) => {
        setUploadProgress(progress);
      });
      
      // Notify parent of successful upload
      onUploadComplete({ 
        imageId, 
        filename: file.name, 
        previewUrl: URL.createObjectURL(file) 
      });
    } catch (err) {
      setError(err.message || 'Failed to upload image');
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.tiff']
    },
    maxSize: 25 * 1024 * 1024, // 25MB limit
    maxFiles: 1
  });
  
  return (
    <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
      {!previewImage ? (
        <Box 
          {...getRootProps()} 
          sx={{
            border: '2px dashed #cccccc',
            borderRadius: 2,
            p: 5,
            textAlign: 'center',
            backgroundColor: isDragActive ? 'rgba(0, 0, 0, 0.05)' : 'transparent',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6">
            {isDragActive ? 'Drop the fabric image here' : 'Drag & drop a fabric image, or click to select'}
          </Typography>
          <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
            Supported formats: JPEG, PNG, TIFF (max 25MB)
          </Typography>
        </Box>
      ) : (
        <Box>
          <ImagePreview 
            imageUrl={previewImage} 
            onRemove={() => setPreviewImage(null)} 
          />
          
          {uploading && (
            <Box display="flex" alignItems="center" mt={2}>
              <CircularProgress 
                variant="determinate" 
                value={uploadProgress} 
                size={24} 
                sx={{ mr: 2 }} 
              />
              <Typography variant="body2">
                Uploading... {Math.round(uploadProgress)}%
              </Typography>
            </Box>
          )}
          
          {error && (
            <Typography color="error" variant="body2" mt={2}>
              Error: {error}
            </Typography>
          )}
        </Box>
      )}
    </Paper>
  );
};

export default ImageUploader;
```

### ThreadVisualizer Component

The ThreadVisualizer component displays the thread detection results with interactive overlays:

```jsx
import React, { useState, useRef, useEffect } from 'react';
import { Box, Paper, ToggleButtonGroup, ToggleButton, Typography, Slider } from '@mui/material';

const ThreadVisualizer = ({ 
  originalImageUrl, 
  warpThreadsUrl, 
  weftThreadsUrl,
  threadCount 
}) => {
  const [viewMode, setViewMode] = useState('combined');
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const canvasRef = useRef(null);
  
  // Images for rendering
  const [images, setImages] = useState({
    original: null,
    warp: null,
    weft: null
  });
  
  // Load all images
  useEffect(() => {
    const loadImage = (url) => {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = url;
      });
    };
    
    const loadAllImages = async () => {
      const [original, warp, weft] = await Promise.all([
        loadImage(originalImageUrl),
        loadImage(warpThreadsUrl),
        loadImage(weftThreadsUrl)
      ]);
      
      setImages({ original, warp, weft });
    };
    
    loadAllImages();
  }, [originalImageUrl, warpThreadsUrl, weftThreadsUrl]);
  
  // Handle view mode change
  const handleViewModeChange = (event, newMode) => {
    if (newMode !== null) {
      setViewMode(newMode);
    }
  };
  
  // Draw the visualization based on current mode
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !images.original) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = images.original;
    
    // Set canvas dimensions to match image
    canvas.width = width;
    canvas.height = height;
    
    // Draw original image
    ctx.drawImage(images.original, 0, 0);
    
    // Apply overlays based on view mode
    if (viewMode === 'combined' || viewMode === 'warp') {
      ctx.globalAlpha = overlayOpacity;
      ctx.drawImage(images.warp, 0, 0);
    }
    
    if (viewMode === 'combined' || viewMode === 'weft') {
      ctx.globalAlpha = overlayOpacity;
      ctx.drawImage(images.weft, 0, 0);
    }
    
    // Reset alpha
    ctx.globalAlpha = 1.0;
    
  }, [images, viewMode, overlayOpacity]);
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Thread Visualization
      </Typography>
      
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={handleViewModeChange}
          size="small"
        >
          <ToggleButton value="original">Original</ToggleButton>
          <ToggleButton value="warp">Warp Threads</ToggleButton>
          <ToggleButton value="weft">Weft Threads</ToggleButton>
          <ToggleButton value="combined">Combined</ToggleButton>
        </ToggleButtonGroup>
        
        <Box sx={{ width: 200, display: 'flex', alignItems: 'center' }}>
          <Typography variant="body2" sx={{ mr: 2 }}>
            Opacity:
          </Typography>
          <Slider
            value={overlayOpacity}
            onChange={(e, newValue) => setOverlayOpacity(newValue)}
            min={0}
            max={1}
            step={0.01}
            size="small"
          />
        </Box>
      </Box>
      
      <Box sx={{ 
        position: 'relative',
        backgroundColor: '#f5f5f5',
        borderRadius: 1,
        overflow: 'hidden',
        textAlign: 'center'
      }}>
        <canvas 
          ref={canvasRef}
          style={{ 
            maxWidth: '100%', 
            height: 'auto',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}
        />
        
        <Box sx={{ 
          position: 'absolute',
          bottom: 16,
          right: 16,
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: 'white',
          borderRadius: 1,
          p: 1
        }}>
          <Typography variant="body2">
            Warp: {threadCount?.warp || '—'} | Weft: {threadCount?.weft || '—'} | TC: {threadCount?.total || '—'}
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ThreadVisualizer;
```

### ResultsCard Component

The ResultsCard displays thread count analysis with charts and metrics:

```jsx
import React from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Divider, 
  Grid, 
  Chip 
} from '@mui/material';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

const getQualityLabel = (threadCount) => {
  if (!threadCount?.total) return 'Unknown';
  
  if (threadCount.total >= 500) return 'Ultra Premium';
  if (threadCount.total >= 300) return 'Premium';
  if (threadCount.total >= 180) return 'Good';
  return 'Basic';
};

const getQualityColor = (quality) => {
  switch (quality) {
    case 'Ultra Premium': return '#8884d8';
    case 'Premium': return '#82ca9d';
    case 'Good': return '#ffc658';
    case 'Basic': return '#ff8042';
    default: return '#cccccc';
  }
};

const ResultsCard = ({ threadCount, analysisDetails }) => {
  const quality = getQualityLabel(threadCount);
  const qualityColor = getQualityColor(quality);
  
  const chartData = [
    { name: 'Warp', threads: threadCount?.warp || 0, fill: '#8884d8' },
    { name: 'Weft', threads: threadCount?.weft || 0, fill: '#82ca9d' },
  ];
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Thread Count Analysis
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Box sx={{ height: 250 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="threads" name="Thread Count" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box>
            <Typography variant="subtitle2" color="textSecondary">
              Thread Count
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Box>
                <Typography variant="h4" fontWeight="bold">
                  {threadCount?.total || '—'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  threads per inch
                </Typography>
              </Box>
              <Chip 
                label={quality} 
                sx={{ 
                  backgroundColor: qualityColor, 
                  color: 'white', 
                  fontWeight: 'bold' 
                }} 
              />
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="subtitle2" color="textSecondary">
                  Warp Threads
                </Typography>
                <Typography variant="h6">
                  {threadCount?.warp || '—'}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="subtitle2" color="textSecondary">
                  Weft Threads
                </Typography>
                <Typography variant="h6">
                  {threadCount?.weft || '—'}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="subtitle2" color="textSecondary">
                  Warp Spacing
                </Typography>
                <Typography variant="body1">
                  {analysisDetails?.warp?.mean_spacing?.toFixed(4) || '—'} in
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="subtitle2" color="textSecondary">
                  Weft Spacing
                </Typography>
                <Typography variant="body1">
                  {analysisDetails?.weft?.mean_spacing?.toFixed(4) || '—'} in
                </Typography>
              </Grid>
            </Grid>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="subtitle2" color="textSecondary">
              Confidence Score
            </Typography>
            <Typography variant="body1">
              {(threadCount?.confidence * 100).toFixed(1) || '—'}%
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default ResultsCard;
```

## State Management

The application uses a combination of React Query for server state and Zustand for client state:

### Thread Analysis Store

```jsx
// store/analysisStore.js
import create from 'zustand';

const useAnalysisStore = create((set, get) => ({
  // Currently uploaded image
  currentImage: null,
  
  // Analysis results
  results: null,
  
  // Processing state
  isProcessing: false,
  processingProgress: 0,
  error: null,
  
  // Set current image
  setCurrentImage: (imageData) => set({ currentImage: imageData }),
  
  // Start processing
  startProcessing: () => set({ 
    isProcessing: true, 
    processingProgress: 0,
    error: null
  }),
  
  // Update processing progress
  updateProgress: (progress) => set({ processingProgress: progress }),
  
  // Set results
  setResults: (results) => set({ 
    results,
    isProcessing: false,
    processingProgress: 100
  }),
  
  // Set error
  setError: (error) => set({ 
    error, 
    isProcessing: false 
  }),
  
  // Clear all data
  clearAll: () => set({ 
    currentImage: null,
    results: null,
    isProcessing: false,
    processingProgress: 0,
    error: null
  }),
}));

export default useAnalysisStore;
```

### API Integration

The frontend integrates with the FastAPI backend through a service layer:

```jsx
// services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Upload image
export const uploadImage = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('image', file);
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      onProgress(percentCompleted);
    },
  });
  
  return response.data.image_id;
};

// Process image and get thread count
export const analyzeImage = async (imageId, options = {}) => {
  const response = await api.post('/predict', {
    image_id: imageId,
    options
  });
  
  return response.data;
};

// Get analysis status
export const getAnalysisStatus = async (imageId) => {
  const response = await api.get(`/status/${imageId}`);
  return response.data;
};

// Get analysis results
export const getResults = async (imageId) => {
  const response = await api.get(`/results/${imageId}`);
  return response.data;
};

export default api;
```

## Main Application Page

The ResultsPage combines these components to display the thread counting results:

```jsx
// pages/ResultsPage.jsx
import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Typography, Box, Button, Grid, Alert } from '@mui/material';
import { ArrowBack, Download } from '@mui/icons-material';
import { useQuery } from 'react-query';

import ThreadVisualizer from '../components/analysis/ThreadVisualizer';
import ResultsCard from '../components/analysis/ResultsCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { getResults } from '../services/api';

const ResultsPage = () => {
  const { imageId } = useParams();
  const navigate = useNavigate();
  
  // Fetch analysis results
  const { data: results, isLoading, error } = useQuery(
    ['analysisResults', imageId],
    () => getResults(imageId),
    {
      enabled: !!imageId,
      refetchOnWindowFocus: false
    }
  );
  
  // Handle download of results
  const handleDownloadResults = () => {
    if (!results) return;
    
    const resultsBlob = new Blob(
      [JSON.stringify(results, null, 2)], 
      { type: 'application/json' }
    );
    
    const url = URL.createObjectURL(resultsBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `thread-analysis-${imageId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  if (isLoading) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <LoadingSpinner message="Loading analysis results..." />
      </Container>
    );
  }
  
  if (error) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          Error loading results: {error.message}
        </Alert>
        <Button 
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          variant="contained"
        >
          Back to Upload
        </Button>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Button 
            startIcon={<ArrowBack />}
            onClick={() => navigate('/')}
            variant="outlined"
            sx={{ mr: 2 }}
          >
            Back
          </Button>
          
          <Typography variant="h4" component="h1" display="inline">
            Thread Analysis Results
          </Typography>
        </Box>
        
        <Button 
          startIcon={<Download />}
          onClick={handleDownloadResults}
          variant="contained"
        >
          Download Results
        </Button>
      </Box>
      
      {results && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <ThreadVisualizer 
              originalImageUrl={results.visualizations.original}
              warpThreadsUrl={results.visualizations.warp_detection}
              weftThreadsUrl={results.visualizations.weft_detection}
              threadCount={results.thread_count}
            />
          </Grid>
          
          <Grid item xs={12}>
            <ResultsCard 
              threadCount={results.thread_count}
              analysisDetails={results.analysis_details}
            />
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default ResultsPage;
```

## User Interface Design

The frontend implements a clean, intuitive interface focused on:

### 1. Simple Upload Process
- Drag-and-drop interface
- Support for multiple image formats
- Progress indication
- Immediate image preview

### 2. Interactive Results
- Visual thread overlay with customizable views
- Color-coded thread highlighting
- Zooming and panning capabilities
- Measurement tool for detailed inspection

### 3. Analysis Dashboard
- Thread count metrics
- Quality assessment
- Comparison with industry standards
- Downloadable reports

### 4. Responsive Design
- Mobile-friendly layout
- Adaptive components
- Touch interactions for mobile users
- Progressive loading for slower connections

## Error Handling

The application implements comprehensive error handling:

```jsx
// components/common/ErrorBoundary.jsx
import React from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error to monitoring service
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Paper 
          elevation={3} 
          sx={{ 
            p: 4, 
            m: 2, 
            textAlign: 'center', 
            backgroundColor: '#fff9f9' 
          }}
        >
          <ErrorOutlineIcon sx={{ fontSize: 60, color: 'error.main', mb: 2 }} />
          <Typography variant="h5" component="h2" gutterBottom>
            Something went wrong
          </Typography>
          <Typography variant="body1" color="textSecondary" paragraph>
            The application encountered an unexpected error. Please try refreshing the page.
          </Typography>
          <Box sx={{ mt: 3 }}>
            <Button 
              variant="contained" 
              color="primary"
              onClick={() => window.location.reload()}
            >
              Refresh Page
            </Button>
          </Box>
        </Paper>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

## Accessibility Considerations

The frontend implements several accessibility features:

- **Keyboard navigation**: All interactive elements are keyboard accessible
- **Screen reader support**: Proper ARIA labels and semantic HTML
- **Color contrast**: All text meets WCAG AA standards
- **Focus indicators**: Visible focus states for keyboard users
- **Error identification**: Clear error messages with instructions

## Performance Optimization

Several techniques are used to ensure optimal frontend performance:

- **Code splitting**: Dynamic imports for route-based code splitting
- **Lazy loading**: Images and components loaded only when needed
- **Memoization**: React.memo and useMemo to prevent unnecessary re-renders
- **Virtualization**: For rendering large lists efficiently
- **Asset optimization**: Compressed images and SVG icons

## Testing Strategy

The frontend implements a comprehensive testing strategy:

```jsx
// Example test for ImageUploader component
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ImageUploader from './ImageUploader';
import { uploadImage } from '../../services/api';

// Mock the API module
jest.mock('../../services/api', () => ({
  uploadImage: jest.fn()
}));

describe('ImageUploader Component', () => {
  const mockFile = new File(['dummy content'], 'test.png', { type: 'image/png' });
  const mockOnUploadComplete = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  test('renders dropzone with correct text', () => {
    render(<ImageUploader onUploadComplete={mockOnUploadComplete} />);
    
    expect(screen.getByText(/Drag & drop a fabric image/i)).toBeInTheDocument();
    expect(screen.getByText(/Supported formats/i)).toBeInTheDocument();
  });
  
  test('shows preview when file is uploaded', async () => {
    // Mock URL.createObjectURL
    URL.createObjectURL = jest.fn(() => 'mock-url');
    
    render(<ImageUploader onUploadComplete={mockOnUploadComplete} />);
    
    // Get dropzone element
    const dropzone = screen.getByText(/Drag & drop a fabric image/i).closest('div');
    
    // Simulate file drop
    await userEvent.upload(
      dropzone.querySelector('input[type="file"]'),
      mockFile
    );
    
    // Preview should be shown
    expect(URL.createObjectURL).toHaveBeenCalledWith(mockFile);
  });
  
  test('calls API and onUploadComplete when file is uploaded', async () => {
    // Mock successful upload
    uploadImage.mockResolvedValue('mock-image-id');
    
    render(<ImageUploader onUploadComplete={mockOnUploadComplete} />);
    
    // Get dropzone input
    const input = screen.getByText(/Drag & drop a fabric image/i)
      .closest('div')
      .querySelector('input');
    
    // Upload file
    fireEvent.change(input, { target: { files: [mockFile] } });
    
    // Check API called
    expect(uploadImage).toHaveBeenCalledWith(
      mockFile,
      expect.any(Function)
    );
    
    // Check onUploadComplete called with correct data
    await waitFor(() => {
      expect(mockOnUploadComplete).toHaveBeenCalledWith({
        imageId: 'mock-image-id',
        filename: 'test.png',
        previewUrl: expect.any(String)
      });
    });
  });
});
```

---

For more detailed technical specifications and implementation guides, refer to the frontend project repository and code documentation.
