import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ResultDisplay from '../../components/ResultDisplay';

// Mock Chart.js
jest.mock('react-chartjs-2', () => ({
  Bar: () => <div data-testid="bar-chart" />,
  Line: () => <div data-testid="line-chart" />
}));

// Sample test data
const mockResults = {
  original_image_path: 'uploads/original.jpg',
  annotated_image_path: 'uploads/annotated.jpg',
  processing_time: 2.5,
  results: {
    warp_count: 32,
    weft_count: 28,
    total_count: 60,
    density: 38.5,
    unit: 'cm'
  }
};

describe('ResultDisplay Component', () => {
  test('renders loading state correctly', () => {
    render(<ResultDisplay loading={true} />);
    expect(screen.getByText('Processing Thread Analysis')).toBeInTheDocument();
    expect(screen.getByText('Please wait while we analyze the fabric image...')).toBeInTheDocument();
  });

  test('renders error state correctly', () => {
    render(<ResultDisplay error="Analysis failed due to invalid image" />);
    expect(screen.getByText('Analysis Error')).toBeInTheDocument();
    expect(screen.getByText('Analysis failed due to invalid image')).toBeInTheDocument();
  });

  test('renders analysis results correctly', () => {
    render(<ResultDisplay results={mockResults} />);
    
    // Check if metrics are displayed
    expect(screen.getByText('Thread Density Analysis Results')).toBeInTheDocument();
    expect(screen.getByText('Thread Count Metrics')).toBeInTheDocument();
    expect(screen.getByText(`${mockResults.results.density} threads/${mockResults.results.unit}²`)).toBeInTheDocument();
    expect(screen.getByText(mockResults.results.warp_count.toString())).toBeInTheDocument();
    expect(screen.getByText(mockResults.results.weft_count.toString())).toBeInTheDocument();
    expect(screen.getByText(mockResults.results.total_count.toString())).toBeInTheDocument();
    
    // Check for processing time
    expect(screen.getByText(`Processing time: ${mockResults.processing_time.toFixed(2)} seconds`)).toBeInTheDocument();
  });

  test('tab switching works correctly', () => {
    render(<ResultDisplay results={mockResults} />);
    
    // Initially, we're on the Metrics tab
    expect(screen.getByText('Thread Count Metrics')).toBeInTheDocument();
    
    // Switch to the Visualization tab
    fireEvent.click(screen.getByText('Visualization'));
    
    // Check that visualization tab content is now visible
    expect(screen.getByText('Thread Count Distribution')).toBeInTheDocument();
    expect(screen.getByText('Fabric Image Analysis')).toBeInTheDocument();
  });

  test('image view switching works correctly', () => {
    render(<ResultDisplay results={mockResults} />);
    
    // Default should be annotated view
    const initialImageSrc = screen.getByAltText('Annotated Fabric Image').getAttribute('src');
    expect(initialImageSrc).toContain('annotated.jpg');
    
    // Switch to original view
    fireEvent.click(screen.getByText('Original'));
    
    // Now should display original image
    const updatedImageSrc = screen.getByAltText('Original Fabric Image').getAttribute('src');
    expect(updatedImageSrc).toContain('original.jpg');
  });

  test('handles null results gracefully', () => {
    const { container } = render(<ResultDisplay results={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  test('download button triggers download function', () => {
    // Mock createElement and other DOM methods
    const mockAnchor = {
      href: null,
      download: null,
      click: jest.fn(),
    };
    
    document.createElement = jest.fn().mockImplementation((tag) => {
      if (tag === 'a') return mockAnchor;
      return document.createElement(tag);
    });
    
    document.body.appendChild = jest.fn();
    document.body.removeChild = jest.fn();
    
    render(<ResultDisplay results={mockResults} />);
    
    // Find and click the download button
    fireEvent.click(screen.getByText('Download'));
    
    // Verify that the download was triggered
    expect(mockAnchor.click).toHaveBeenCalledTimes(1);
    expect(document.body.appendChild).toHaveBeenCalledWith(mockAnchor);
    expect(document.body.removeChild).toHaveBeenCalledWith(mockAnchor);
  });
});
