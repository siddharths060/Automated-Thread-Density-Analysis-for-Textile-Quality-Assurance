import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import AnalyzeButton from '../../components/AnalyzeButton';
import ThreadAnalysisService from '../../services/ThreadAnalysisService';

// Mock the ThreadAnalysisService
jest.mock('../../services/ThreadAnalysisService', () => ({
  analyzeImage: jest.fn()
}));

describe('AnalyzeButton Component', () => {
  const mockFile = new File(['dummy content'], 'test-fabric.png', { type: 'image/png' });
  const mockSetResults = jest.fn();
  const mockSetError = jest.fn();
  const mockSetLoading = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  test('renders disabled button when no file is selected', () => {
    render(
      <AnalyzeButton 
        file={null} 
        setResults={mockSetResults} 
        setError={mockSetError} 
        setLoading={mockSetLoading} 
      />
    );
    
    const button = screen.getByRole('button', { name: /analyze fabric/i });
    expect(button).toBeDisabled();
  });
  
  test('renders enabled button when file is selected', () => {
    render(
      <AnalyzeButton 
        file={mockFile} 
        setResults={mockSetResults} 
        setError={mockSetError} 
        setLoading={mockSetLoading} 
      />
    );
    
    const button = screen.getByRole('button', { name: /analyze fabric/i });
    expect(button).toBeEnabled();
  });
  
  test('calls analyzeImage and updates state on successful analysis', async () => {
    const mockResults = {
      success: true,
      message: 'Analysis completed successfully',
      original_image_path: 'uploads/original.jpg',
      annotated_image_path: 'uploads/annotated.jpg',
      results: {
        warp_count: 30,
        weft_count: 25,
        total_count: 55,
        density: 35.5,
        unit: 'cm'
      },
      processing_time: 2.5
    };
    
    ThreadAnalysisService.analyzeImage.mockResolvedValue(mockResults);
    
    render(
      <AnalyzeButton 
        file={mockFile} 
        setResults={mockSetResults} 
        setError={mockSetError} 
        setLoading={mockSetLoading} 
      />
    );
    
    const button = screen.getByRole('button', { name: /analyze fabric/i });
    fireEvent.click(button);
    
    // Check if loading state is set to true initially
    expect(mockSetLoading).toHaveBeenCalledWith(true);
    
    await waitFor(() => {
      // Verify service was called with the file
      expect(ThreadAnalysisService.analyzeImage).toHaveBeenCalledWith(mockFile, expect.any(Function));
      
      // Check if results are set correctly
      expect(mockSetResults).toHaveBeenCalledWith(mockResults);
      
      // Check if loading state is set back to false
      expect(mockSetLoading).toHaveBeenCalledWith(false);
      
      // Check if error state is cleared
      expect(mockSetError).toHaveBeenCalledWith(null);
    });
  });
  
  test('handles error correctly during analysis', async () => {
    const mockError = new Error('Failed to analyze image');
    ThreadAnalysisService.analyzeImage.mockRejectedValue(mockError);
    
    render(
      <AnalyzeButton 
        file={mockFile} 
        setResults={mockSetResults} 
        setError={mockSetError} 
        setLoading={mockSetLoading} 
      />
    );
    
    const button = screen.getByRole('button', { name: /analyze fabric/i });
    fireEvent.click(button);
    
    await waitFor(() => {
      // Verify error state is updated
      expect(mockSetError).toHaveBeenCalledWith('Failed to analyze image');
      
      // Verify loading state is set back to false
      expect(mockSetLoading).toHaveBeenCalledWith(false);
      
      // Verify results are not set
      expect(mockSetResults).not.toHaveBeenCalled();
    });
  });
  
  test('shows progress indicator during file upload', async () => {
    // Mock implementation that calls the onProgress callback
    ThreadAnalysisService.analyzeImage.mockImplementation((file, onProgress) => {
      // Simulate progress events
      onProgress(25);
      onProgress(50);
      onProgress(75);
      onProgress(100);
      
      return Promise.resolve({
        success: true,
        message: 'Analysis completed successfully'
      });
    });
    
    render(
      <AnalyzeButton 
        file={mockFile} 
        setResults={mockSetResults} 
        setError={mockSetError} 
        setLoading={mockSetLoading} 
      />
    );
    
    const button = screen.getByRole('button', { name: /analyze fabric/i });
    fireEvent.click(button);
    
    // The progress indicators should be visible
    expect(await screen.findByText('Processing...')).toBeInTheDocument();
  });
});
