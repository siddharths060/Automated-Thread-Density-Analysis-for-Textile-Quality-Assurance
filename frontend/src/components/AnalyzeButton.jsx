import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  CircularProgress,
  Alert,
  Paper,
  Fade,
  LinearProgress,
  Tooltip,
  Divider,
  Chip
} from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import InfoIcon from '@mui/icons-material/Info';
import SettingsSuggestIcon from '@mui/icons-material/SettingsSuggest';
import GridOnIcon from '@mui/icons-material/GridOn';

import ThreadAnalysisService from '../services/ThreadAnalysisService';

const ANALYSIS_STEPS = [
  { label: 'Preparing Image', time: 1000 },
  { label: 'Detecting Threads', time: 2500 },
  { label: 'Counting Warp Threads', time: 1500 },
  { label: 'Counting Weft Threads', time: 1500 },
  { label: 'Calculating Density', time: 1000 },
  { label: 'Generating Visualization', time: 2000 }
];

const AnalyzeButton = ({ 
  imagePath, 
  onAnalysisStart, 
  onAnalysisComplete,
  isLoading
}) => {
  const [error, setError] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [showTip, setShowTip] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  
  // Simulate processing steps when analysis is running
  useEffect(() => {
    let stepTimer;
    if (isLoading && currentStep < ANALYSIS_STEPS.length) {
      stepTimer = setTimeout(() => {
        setCurrentStep(prevStep => Math.min(prevStep + 1, ANALYSIS_STEPS.length - 1));
      }, ANALYSIS_STEPS[currentStep].time);
    }
    
    return () => {
      if (stepTimer) clearTimeout(stepTimer);
    };
  }, [isLoading, currentStep]);
  
  // Reset the steps when not loading
  useEffect(() => {
    if (!isLoading) {
      setCurrentStep(0);
    }
  }, [isLoading]);
  
  // Show a tip after 5 seconds of loading
  useEffect(() => {
    let tipTimer;
    if (isLoading) {
      tipTimer = setTimeout(() => {
        setShowTip(true);
      }, 5000);
    } else {
      setShowTip(false);
    }
    
    return () => {
      if (tipTimer) clearTimeout(tipTimer);
    };
  }, [isLoading]);
  
  const handleAnalyze = async () => {
    setError(null);
    onAnalysisStart();
    
    try {
      const results = await ThreadAnalysisService.analyzeImage(imagePath);
      if (results.success) {
        onAnalysisComplete(results);
      } else {
        throw new Error(results.error || 'Analysis failed');
      }
    } catch (err) {
      console.error("Analysis error:", err);
      setError(
        err.customMessage || err.response?.data?.detail || err.message || 
        'Error analyzing image. Please try again.'
      );
      onAnalysisComplete(null);
      setRetryCount(prev => prev + 1);
    }
  };
  
  return (
    <Box sx={{ textAlign: 'center' }}>
      <Paper
        sx={{ 
          p: 3, 
          mb: 3, 
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
      >
        <Typography variant="h6" gutterBottom>
          Ready to Analyze
        </Typography>
        
        <Typography variant="body1" sx={{ mb: 3 }} color="textSecondary">
          The image has been uploaded successfully. Click the button below to analyze thread density.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3, width: '100%' }}>
            {error}
          </Alert>
        )}
        
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={isLoading ? <CircularProgress size={24} color="inherit" /> : <AnalyticsIcon />}
          onClick={handleAnalyze}
          disabled={isLoading}
          sx={{ py: 1.5, px: 4 }}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Thread Density'}
        </Button>
        
        {isLoading && (
          <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
            This may take a few moments depending on the image size and complexity
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default AnalyzeButton;
