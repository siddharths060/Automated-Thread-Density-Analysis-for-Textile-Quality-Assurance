import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import UploadImage from '../../components/UploadImage';

// Mock React Dropzone
jest.mock('react-dropzone', () => ({
  __esModule: true,
  default: ({ onDrop, children }) => {
    const handleClick = () => {
      const fileInput = document.createElement('input');
      fileInput.type = 'file';
      fileInput.addEventListener('change', (e) => {
        onDrop(e.target.files);
      });
      fileInput.click();
    };

    return (
      <div data-testid="dropzone" onClick={handleClick}>
        {children({ 
          isDragActive: false, 
          isDragAccept: false, 
          isDragReject: false, 
          getRootProps: () => ({ onClick: handleClick }), 
          getInputProps: () => ({}) 
        })}
      </div>
    );
  }
}));

describe('UploadImage Component', () => {
  const mockOnFileUpload = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders upload area correctly', () => {
    render(<UploadImage onFileUpload={mockOnFileUpload} />);
    
    expect(screen.getByText('Drag & Drop Fabric Image Here')).toBeInTheDocument();
    expect(screen.getByText('or click to select')).toBeInTheDocument();
  });

  test('calls onFileUpload when file is selected', async () => {
    render(<UploadImage onFileUpload={mockOnFileUpload} />);
    
    const file = new File(['dummy content'], 'test.png', { type: 'image/png' });
    
    // Create a mock dropzone event
    const dropzone = screen.getByTestId('dropzone');
    
    // Mock the FileList
    const dataTransfer = {
      files: [file],
    };
    
    // Trigger file drop
    fireEvent.drop(dropzone, { dataTransfer });
    
    // Wait for the onFileUpload to be called
    await waitFor(() => {
      expect(mockOnFileUpload).toHaveBeenCalledTimes(1);
      expect(mockOnFileUpload).toHaveBeenCalledWith(file);
    });
  });
  
  test('displays error for invalid file type', async () => {
    render(<UploadImage onFileUpload={mockOnFileUpload} />);
    
    const file = new File(['dummy content'], 'test.pdf', { type: 'application/pdf' });
    
    // Create a mock dropzone event
    const dropzone = screen.getByTestId('dropzone');
    
    // Mock the FileList
    const dataTransfer = {
      files: [file],
    };
    
    // Trigger file drop
    fireEvent.drop(dropzone, { dataTransfer });
    
    // Verify error message is displayed
    await waitFor(() => {
      expect(screen.getByText(/Invalid file type/i)).toBeInTheDocument();
    });
    
    // Verify onFileUpload was not called
    expect(mockOnFileUpload).not.toHaveBeenCalled();
  });
  
  test('displays error for file size limit', async () => {
    render(<UploadImage onFileUpload={mockOnFileUpload} />);
    
    // Create a file with size above the limit
    const largeFile = new File([new ArrayBuffer(26 * 1024 * 1024)], 'large.jpg', { type: 'image/jpeg' });
    
    // Create a mock dropzone event
    const dropzone = screen.getByTestId('dropzone');
    
    // Mock the FileList
    const dataTransfer = {
      files: [largeFile],
    };
    
    // Trigger file drop
    fireEvent.drop(dropzone, { dataTransfer });
    
    // Verify error message is displayed
    await waitFor(() => {
      expect(screen.getByText(/File size exceeds/i)).toBeInTheDocument();
    });
    
    // Verify onFileUpload was not called
    expect(mockOnFileUpload).not.toHaveBeenCalled();
  });

  test('displays file preview after selection', async () => {
    render(<UploadImage onFileUpload={mockOnFileUpload} />);
    
    // Create a test image file
    const file = new File(['dummy content'], 'test.png', { type: 'image/png' });
    
    // Mock URL.createObjectURL
    const mockUrl = 'blob:mockedurl';
    global.URL.createObjectURL = jest.fn(() => mockUrl);
    
    // Create a mock dropzone event
    const dropzone = screen.getByTestId('dropzone');
    
    // Mock the FileList
    const dataTransfer = {
      files: [file],
    };
    
    // Trigger file drop
    fireEvent.drop(dropzone, { dataTransfer });
    
    // Wait for the preview to be displayed
    await waitFor(() => {
      const previewImage = screen.getByAltText('Uploaded Preview');
      expect(previewImage).toBeInTheDocument();
      expect(previewImage.src).toBe(mockUrl);
      expect(screen.getByText('test.png')).toBeInTheDocument();
    });
    
    // Clean up
    global.URL.createObjectURL.mockRestore();
  });
});
