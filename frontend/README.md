# Thread Density Analyzer Frontend

This is the frontend application for the Automated Thread Density Analysis Tool. It provides a user interface for uploading fabric images and displaying thread count analysis results.

## Features

- Image upload via drag & drop or file selector
- Thread density analysis visualization
- Result display with detailed metrics
- Interactive chart visualization

## Technology Stack

- React.js 18
- Material UI for component styling
- Chart.js for data visualization
- Axios for API communication

## Setup Instructions

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Project Structure

- `src/components/`: React components
  - `UploadImage.jsx`: Component for image upload
  - `AnalyzeButton.jsx`: Component for triggering analysis
  - `ResultDisplay.jsx`: Component for displaying results
- `src/services/`: API services
  - `ThreadAnalysisService.js`: Service for communicating with backend API

## API Integration

The frontend communicates with the backend API at `http://localhost:8000/api`. The following endpoints are used:

- `POST /api/analysis/upload`: Upload an image file
- `POST /api/analysis/predict`: Analyze thread density in an uploaded image

## Troubleshooting

- If the API URL needs to be changed, you can set the `REACT_APP_API_URL` environment variable.
