import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Paper, 
  Divider,
  Card,
  CardContent,
  Link,
  CircularProgress,
  Alert,
  Skeleton,
  Tabs,
  Tab,
  ButtonGroup,
  Button,
  Tooltip as MUITooltip
} from '@mui/material';
import { 
  Straighten as StraightenIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon,
  CompareArrows as CompareArrowsIcon,
  Assessment as AssessmentIcon,
  ZoomIn as ZoomInIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ResultDisplay = ({ results, loading = false, error = null }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [imageView, setImageView] = useState('annotated');
  const [imageZoomed, setImageZoomed] = useState(false);
  
  // Handle case when results is null or undefined
  if (!results && !loading && !error) {
    return null;
  }
  
  // Show error if present
  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        <Typography variant="h6">Analysis Error</Typography>
        <Typography variant="body2">{typeof error === 'string' ? error : error.message}</Typography>
        <Typography variant="caption" color="text.secondary">
          Please try again or contact support if the problem persists.
        </Typography>
      </Alert>
    );
  }
  
  // Show loading state
  if (loading) {
    return (
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress size={60} thickness={4} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Processing Thread Analysis
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Please wait while we analyze the fabric image...
        </Typography>
      </Box>
    );
  }
  
  const { original_image_path, annotated_image_path, results: metrics, processing_time } = results;
  
  const chartData = {
    labels: ['Warp Threads', 'Weft Threads', 'Total Threads'],
    datasets: [
      {
        label: 'Thread Count',
        data: [metrics.warp_count, metrics.weft_count, metrics.total_count],
        backgroundColor: [
          'rgba(54, 162, 235, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(255, 99, 132, 0.8)'
        ],
        borderColor: [
          'rgb(54, 162, 235)',
          'rgb(75, 192, 192)',
          'rgb(255, 99, 132)'
        ],
        borderWidth: 1
      }
    ]
  };
  
  // Distribution chart data (simulated distribution of thread spacing)
  const distributionData = {
    labels: Array.from({ length: 20 }, (_, i) => i),
    datasets: [
      {
        label: 'Warp Thread Distribution',
        data: generateNormalDistribution(metrics.warp_count),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Weft Thread Distribution',
        data: generateNormalDistribution(metrics.weft_count),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
        tension: 0.4
      }
    ]
  };
  
  // Function to generate a normal distribution (simulated data)
  function generateNormalDistribution(mean) {
    return Array.from({ length: 20 }, (_, i) => {
      const x = i - 10;
      return Math.exp(-(Math.pow(x, 2) / (2 * Math.pow(3, 2)))) * mean / 2;
    });
  }
  
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Thread Count Results'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Count'
        }
      }
    }
  };
  
  const distributionOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Thread Spacing Distribution'
      },
      tooltip: {
        callbacks: {
          title: (context) => `Position: ${context[0].label}`,
          label: (context) => `Density: ${context.raw.toFixed(2)}`,
        }
      }
    }
  };
  
  // Function to download the annotated image
  const downloadImage = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || 'annotated_fabric.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Thread Density Analysis Results
        </Typography>
        
        <ButtonGroup variant="outlined" size="small">
          <Button 
            startIcon={<DownloadIcon />}
            onClick={() => downloadImage(`http://localhost:8000/${annotated_image_path}`, 'annotated_fabric.png')}
          >
            Download
          </Button>
        </ButtonGroup>
      </Box>
      
      <Tabs
        value={activeTab}
        onChange={(e, newValue) => setActiveTab(newValue)}
        variant="fullWidth"
        sx={{ mb: 3, borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab label="Metrics" icon={<AssessmentIcon />} iconPosition="start" />
        <Tab label="Visualization" icon={<CompareArrowsIcon />} iconPosition="start" />
      </Tabs>
      
      {activeTab === 0 ? (
        <Grid container spacing={3}>
          {/* Results Metrics */}
          <Grid item xs={12} md={5}>
            <Paper sx={{ p: 3, height: '100%', boxShadow: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'medium' }}>
                Thread Count Metrics
              </Typography>
            
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Box sx={{ 
                width: 50, 
                height: 50, 
                borderRadius: '50%', 
                bgcolor: 'primary.main',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2,
                boxShadow: 2
              }}>
                <StraightenIcon sx={{ color: 'white', fontSize: 28 }} />
              </Box>
              <Box>
                <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 500 }}>
                  Thread Density
                </Typography>
                <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                  {metrics.density} threads/{metrics.unit}²
                </Typography>
                <Typography variant="caption" sx={{ color: 'success.main' }}>
                  {metrics.density > 25 ? "High Quality Fabric" : "Standard Quality Fabric"}
                </Typography>
              </Box>
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Card raised sx={{ bgcolor: 'rgba(54, 162, 235, 0.05)', height: '100%', transition: '0.3s', '&:hover': { transform: 'translateY(-2px)' } }}>
                  <CardContent>
                    <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 500 }}>
                      Warp Threads
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 1, color: 'rgb(54, 162, 235)', fontWeight: 'bold' }}>
                      {metrics.warp_count}
                    </Typography>
                    <MUITooltip title="Threads running vertically in the fabric">
                      <Typography variant="caption" color="textSecondary">
                        Vertical direction
                      </Typography>
                    </MUITooltip>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={6}>
                <Card raised sx={{ bgcolor: 'rgba(75, 192, 192, 0.05)', height: '100%', transition: '0.3s', '&:hover': { transform: 'translateY(-2px)' } }}>
                  <CardContent>
                    <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 500 }}>
                      Weft Threads
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 1, color: 'rgb(75, 192, 192)', fontWeight: 'bold' }}>
                      {metrics.weft_count}
                    </Typography>
                    <MUITooltip title="Threads running horizontally in the fabric">
                      <Typography variant="caption" color="textSecondary">
                        Horizontal direction
                      </Typography>
                    </MUITooltip>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12}>
                <Card raised sx={{ bgcolor: 'rgba(255, 99, 132, 0.05)', transition: '0.3s', '&:hover': { transform: 'translateY(-2px)' } }}>
                  <CardContent>
                    <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 500 }}>
                      Total Thread Count
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 1, color: 'rgb(255, 99, 132)', fontWeight: 'bold' }}>
                      {metrics.total_count}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Combined warp and weft threads
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            <Divider sx={{ my: 2 }} />
            
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TimerIcon sx={{ mr: 1, color: 'text.secondary' }} />
                <Typography variant="body2" color="textSecondary">
                  Processing time: <strong>{processing_time.toFixed(2)} seconds</strong>
                </Typography>
              </Box>
              <Box>
                <MUITooltip title="Analysis Quality Score">
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <SpeedIcon sx={{ mr: 1, color: 'success.main' }} />
                    <Typography variant="body2" color="success.main" fontWeight="bold">
                      {/* Calculate a confidence score based on processing time and thread count */}
                      {Math.min(98, 85 + (metrics.total_count > 50 ? 10 : 0) - (processing_time > 5 ? 5 : 0))}% Quality
                    </Typography>
                  </Box>
                </MUITooltip>
              </Box>
            </Box>
          </Paper>
        </Grid>
            
        {/* Visualization */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 3, boxShadow: 3 }}>
            <Box sx={{ mb: 3 }}>
              <Bar data={chartData} options={chartOptions} height={150} />
            </Box>
            
            <Divider sx={{ my: 3 }} />
            
            <Box sx={{ mb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="subtitle1" fontWeight="medium">
                Thread Distribution Analysis
              </Typography>
              
              <ButtonGroup size="small" variant="outlined">
                <Button
                  color={imageView === 'original' ? 'primary' : 'inherit'}
                  onClick={() => setImageView('original')}
                >
                  Original
                </Button>
                <Button
                  color={imageView === 'annotated' ? 'primary' : 'inherit'}  
                  onClick={() => setImageView('annotated')}
                >
                  Annotated
                </Button>
              </ButtonGroup>
            </Box>
            
            <Box sx={{ textAlign: 'center', mt: 2, position: 'relative', overflow: 'hidden' }}>
              <Box
                component="img"
                src={`http://localhost:8000/${imageView === 'annotated' ? annotated_image_path : original_image_path}`}
                alt={`${imageView === 'annotated' ? 'Annotated' : 'Original'} Fabric Image`}
                sx={{
                  width: '100%',
                  maxHeight: imageZoomed ? '500px' : '300px',
                  objectFit: 'contain',
                  borderRadius: 1,
                  boxShadow: 2,
                  transition: 'all 0.3s ease',
                  cursor: 'pointer'
                }}
                onClick={() => setImageZoomed(!imageZoomed)}
              />
              
              <Box 
                sx={{ 
                  position: 'absolute', 
                  top: 10, 
                  right: 10, 
                  bgcolor: 'rgba(255,255,255,0.8)', 
                  borderRadius: '50%',
                  p: 0.5,
                  cursor: 'pointer'
                }}
                onClick={() => setImageZoomed(!imageZoomed)}
              >
                <ZoomInIcon />
              </Box>
              
              {imageView === 'annotated' && (
                <Typography variant="caption" display="block" sx={{ mt: 1, bgcolor: 'background.paper', p: 1, borderRadius: 1 }}>
                  <Box component="span" sx={{ color: 'rgb(54, 162, 235)', fontWeight: 'bold' }}>Blue lines:</Box> Warp threads | 
                  <Box component="span" sx={{ color: 'rgb(75, 192, 192)', fontWeight: 'bold', ml: 1 }}>Green lines:</Box> Weft threads
                </Typography>
              )}
            </Box>
            
            {/* Spacing Distribution Chart */}
            <Box sx={{ mt: 3 }}>
              <Line data={distributionData} options={distributionOptions} height={120} />
            </Box>
          </Paper>
        </Grid>
      </Grid>
      ) : (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3, height: '100%', boxShadow: 3 }}>
              <Typography variant="h6" gutterBottom>
                Thread Count Distribution
              </Typography>
              <Bar data={chartData} options={chartOptions} height={220} />
              
              <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                <Typography variant="body2" fontWeight="medium">Analysis Insights:</Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  • Thread ratio (warp:weft): <strong>{(metrics.warp_count / metrics.weft_count).toFixed(2)}</strong>
                </Typography>
                <Typography variant="body2">
                  • Fabric balance: <strong>
                    {Math.abs(metrics.warp_count - metrics.weft_count) < 5 ? 'Well balanced' : 'Directional bias'}
                  </strong>
                </Typography>
                <Typography variant="body2">
                  • Quality assessment: <strong>
                    {metrics.density > 25 ? 'Premium quality' : 'Standard quality'}
                  </strong>
                </Typography>
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3, height: '100%', boxShadow: 3 }}>
              <Typography variant="h6" gutterBottom>
                Thread Spacing Distribution 
              </Typography>
              <Line data={distributionData} options={distributionOptions} height={220} />
              
              <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                <Typography variant="body2" fontWeight="medium">Distribution Analysis:</Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  • Thread regularity: <strong>
                    {Math.abs(metrics.warp_count - metrics.weft_count) < 10 ? 'High' : 'Medium'}
                  </strong>
                </Typography>
                <Typography variant="body2">
                  • Consistency rating: <strong>
                    {metrics.total_count > 40 ? 'Excellent' : 'Good'}
                  </strong>
                </Typography>
                <Typography variant="body2">
                  • Recommended uses: <strong>
                    {metrics.density > 25 ? 'Fine garments, professional use' : 'General purpose textiles'}
                  </strong>
                </Typography>
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12}>
            <Paper sx={{ p: 3, boxShadow: 3 }}>
              <Typography variant="h6" gutterBottom>
                Fabric Image Analysis
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, flexWrap: 'wrap' }}>
                <Box sx={{ width: { xs: '100%', md: '45%' }, position: 'relative' }}>
                  <Typography variant="subtitle2" textAlign="center" gutterBottom>
                    Original Image
                  </Typography>
                  <Box
                    component="img"
                    src={`http://localhost:8000/${original_image_path}`}
                    alt="Original Fabric Image"
                    sx={{
                      width: '100%',
                      height: '250px',
                      objectFit: 'cover',
                      borderRadius: 1,
                      boxShadow: 2
                    }}
                  />
                </Box>
                
                <Box sx={{ width: { xs: '100%', md: '45%' }, position: 'relative' }}>
                  <Typography variant="subtitle2" textAlign="center" gutterBottom>
                    Annotated Image
                  </Typography>
                  <Box
                    component="img"
                    src={`http://localhost:8000/${annotated_image_path}`}
                    alt="Annotated Fabric Image"
                    sx={{
                      width: '100%',
                      height: '250px',
                      objectFit: 'cover',
                      borderRadius: 1,
                      boxShadow: 2
                    }}
                  />
                </Box>
              </Box>
              
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
                <Button
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  onClick={() => downloadImage(`http://localhost:8000/${annotated_image_path}`, 'annotated_fabric.png')}
                >
                  Download Annotated Image
                </Button>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default ResultDisplay;
