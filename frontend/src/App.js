import React, { useState } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box,
  Stepper,
  Step,
  StepLabel,
} from '@mui/material';

import UploadImage from './components/UploadImage';
import AnalyzeButton from './components/AnalyzeButton';
import ResultDisplay from './components/ResultDisplay';

function App() {
  const [activeStep, setActiveStep] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadResponse, setUploadResponse] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const steps = ['Upload Image', 'Analyze Threads', 'View Results'];

  const handleFileUpload = (file) => {
    setUploadedFile(file);
    setUploadResponse(null);
    setAnalysisResults(null);
    setActiveStep(1);
  };
  
  const handleUploadSuccess = (response) => {
    setUploadResponse(response);
  };
  
  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setActiveStep(2);
    setIsLoading(false);
  };
  
  const handleAnalysisStart = () => {
    setIsLoading(true);
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography 
        variant="h4" 
        component="h1" 
        gutterBottom 
        align="center" 
        sx={{ mb: 4, fontWeight: 600 }}
      >
        Thread Density Analyzer
      </Typography>
      
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Box sx={{ mt: 2 }}>
          {activeStep === 0 && (
            <UploadImage 
              onFileUpload={handleFileUpload} 
              onUploadSuccess={handleUploadSuccess}
            />
          )}
          
          {activeStep === 1 && uploadResponse && (
            <AnalyzeButton 
              imagePath={uploadResponse.image_path}
              onAnalysisStart={handleAnalysisStart}
              onAnalysisComplete={handleAnalysisComplete}
              isLoading={isLoading}
            />
          )}
          
          {activeStep === 2 && analysisResults && (
            <ResultDisplay results={analysisResults} />
          )}
        </Box>
      </Paper>
      
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="body2" color="textSecondary">
          © 2025 Thread Density Analysis - Automated Quality Assurance for Textile Industry
        </Typography>
      </Box>
    </Container>
  );
}

export default App;
