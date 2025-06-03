import axios from 'axios';
import ThreadAnalysisService from '../../services/ThreadAnalysisService';

// Mock axios
jest.mock('axios');

describe('ThreadAnalysisService', () => {
  const mockFile = new File(['dummy content'], 'test-fabric.png', { type: 'image/png' });
  const mockOnProgress = jest.fn();
  const mockUploadId = { data: { image_path: 'uploads/123.png' } };
  const mockAnalysisResult = { 
    data: {
      success: true,
      message: 'Analysis completed successfully',
      original_image_path: 'uploads/123.png',
      annotated_image_path: 'uploads/123-annotated.png',
      results: {
        warp_count: 30,
        weft_count: 25,
        total_count: 55,
        density: 35.5,
        unit: 'cm'
      },
      processing_time: 2.5
    }
  };
  
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  test('analyzeImage method uploads file and requests analysis', async () => {
    // Mock the upload endpoint
    axios.post.mockImplementation((url, data, config) => {
      if (url === 'http://localhost:8000/api/analysis/upload') {
        return Promise.resolve(mockUploadId);
      } else if (url === 'http://localhost:8000/api/analysis/detect') {
        return Promise.resolve(mockAnalysisResult);
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
    
    const result = await ThreadAnalysisService.analyzeImage(mockFile, mockOnProgress);
    
    // Check that file was uploaded correctly
    expect(axios.post).toHaveBeenCalledWith(
      'http://localhost:8000/api/analysis/upload',
      expect.any(FormData),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'multipart/form-data'
        }),
        onUploadProgress: expect.any(Function)
      })
    );
    
    // Check that analysis was requested
    expect(axios.post).toHaveBeenCalledWith(
      'http://localhost:8000/api/analysis/detect',
      { image_path: 'uploads/123.png' },
      expect.any(Object)
    );
    
    // Verify the result matches the mock data
    expect(result).toEqual(mockAnalysisResult.data);
  });
  
  test('handles error during file upload', async () => {
    const uploadError = new Error('Failed to upload file');
    axios.post.mockRejectedValueOnce(uploadError);
    
    await expect(ThreadAnalysisService.analyzeImage(mockFile, mockOnProgress))
      .rejects
      .toThrow('Failed to upload file');
  });
  
  test('handles error during analysis', async () => {
    // Mock successful upload but failed analysis
    axios.post.mockImplementationOnce(() => Promise.resolve(mockUploadId))
            .mockImplementationOnce(() => Promise.reject(new Error('Analysis failed')));
    
    await expect(ThreadAnalysisService.analyzeImage(mockFile, mockOnProgress))
      .rejects
      .toThrow('Analysis failed');
  });
  
  test('progress callback is invoked correctly', async () => {
    // Mock implementations
    axios.post.mockImplementation((url, data, config) => {
      if (url === 'http://localhost:8000/api/analysis/upload') {
        // Call the onUploadProgress with a mock event
        if (config && config.onUploadProgress) {
          config.onUploadProgress({ loaded: 50, total: 100 });
        }
        return Promise.resolve(mockUploadId);
      } else if (url === 'http://localhost:8000/api/analysis/detect') {
        return Promise.resolve(mockAnalysisResult);
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
    
    await ThreadAnalysisService.analyzeImage(mockFile, mockOnProgress);
    
    // Verify onProgress was called with 50%
    expect(mockOnProgress).toHaveBeenCalledWith(50);
  });
});
