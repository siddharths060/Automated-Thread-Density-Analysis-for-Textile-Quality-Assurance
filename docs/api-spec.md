# 🔌 API Specification

This document details the REST API endpoints provided by the Automated Thread Density Analysis backend service.

## API Base URL

- **Development**: `http://localhost:8000/api`
- **Production**: `https://api.thread-density-analyzer.com/api`

## Authentication

The API uses JWT (JSON Web Token) authentication for secured endpoints:

```
Authorization: Bearer <your_token>
```

Public endpoints can be accessed without authentication.

## Endpoints

### 1. Upload Fabric Image

Upload a fabric image for analysis.

**Endpoint**: `POST /upload`

**Authentication**: Optional

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image | File | Yes | The fabric image file (JPEG, PNG, TIFF) |
| dpi | Number | No | Image DPI if known (default: auto-detect) |
| calibrationSquareSize | Number | No | Size in inches of calibration square if present (default: 1.0) |
| includePreprocessing | Boolean | No | Return preprocessing steps (default: false) |

**Example Request**:

```bash
curl -X POST \
  https://api.thread-density-analyzer.com/api/upload \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@fabric_sample.jpg' \
  -F 'dpi=300' \
  -F 'includePreprocessing=true'
```

**Response**:

```json
{
  "success": true,
  "image_id": "img_12345678",
  "filename": "fabric_sample.jpg",
  "upload_timestamp": "2025-05-29T10:15:30Z",
  "message": "Image uploaded successfully. Use the image_id to request analysis."
}
```

**Status Codes**:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid image format or parameters |
| 401 | Unauthorized - Invalid or missing token |
| 413 | Payload Too Large - Image exceeds size limit |
| 500 | Server Error |

### 2. Process Image & Predict Thread Count

Process an uploaded image and return thread count analysis.

**Endpoint**: `POST /predict`

**Authentication**: Required

**Content-Type**: `application/json`

**Request Body**:

```json
{
  "image_id": "img_12345678",
  "options": {
    "orientation_correction": true,
    "roi": {
      "x": 100,
      "y": 100,
      "width": 500,
      "height": 500
    },
    "return_visualization": true,
    "thread_highlighting": {
      "warp_color": [255, 0, 0],
      "weft_color": [0, 0, 255]
    }
  }
}
```

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_id | String | Yes | ID of the uploaded image |
| options.orientation_correction | Boolean | No | Auto-correct fabric orientation (default: true) |
| options.roi | Object | No | Region of interest for analysis |
| options.return_visualization | Boolean | No | Return annotated image (default: true) |
| options.thread_highlighting | Object | No | Thread visualization colors |

**Example Request**:

```bash
curl -X POST \
  https://api.thread-density-analyzer.com/api/predict \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE' \
  -H 'Content-Type: application/json' \
  -d '{
    "image_id": "img_12345678",
    "options": {
      "orientation_correction": true,
      "return_visualization": true
    }
  }'
```

**Response**:

```json
{
  "success": true,
  "image_id": "img_12345678",
  "thread_count": {
    "warp": 145,
    "weft": 98,
    "total": 243
  },
  "quality_assessment": {
    "grade": "Premium",
    "score": 85,
    "confidence": 0.92
  },
  "analysis_details": {
    "warp": {
      "mean_spacing": 0.0068,
      "std_deviation": 0.0004,
      "threads_per_inch": 145
    },
    "weft": {
      "mean_spacing": 0.0102,
      "std_deviation": 0.0006,
      "threads_per_inch": 98
    },
    "processing_time_ms": 326
  },
  "visualizations": {
    "annotated_image": "data:image/png;base64,iVBOR...",
    "warp_detection": "data:image/png;base64,iVBOR...",
    "weft_detection": "data:image/png;base64,iVBOR..."
  }
}
```

**Status Codes**:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing token |
| 404 | Not Found - Image ID not found |
| 422 | Unprocessable Entity - Image cannot be processed |
| 500 | Server Error |

### 3. Get Processing Status

Check the status of an image processing request.

**Endpoint**: `GET /status/{image_id}`

**Authentication**: Required

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_id | String | Yes | ID of the uploaded image |

**Example Request**:

```bash
curl -X GET \
  https://api.thread-density-analyzer.com/api/status/img_12345678 \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

**Response**:

```json
{
  "image_id": "img_12345678",
  "status": "processing",
  "progress": 75,
  "message": "Thread detection in progress",
  "estimated_time_remaining_seconds": 2
}
```

**Status Values**:

- `queued`: Request is in the processing queue
- `preprocessing`: Image is being preprocessed
- `processing`: Thread detection is in progress
- `completed`: Analysis is complete
- `failed`: Processing failed

**Status Codes**:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 401 | Unauthorized - Invalid or missing token |
| 404 | Not Found - Image ID not found |
| 500 | Server Error |

### 4. Get Analysis Results

Retrieve the results of a completed analysis.

**Endpoint**: `GET /results/{image_id}`

**Authentication**: Required

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_id | String | Yes | ID of the analyzed image |

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| include_visualizations | Boolean | No | Include visualization images (default: true) |
| visualization_format | String | No | Format for visualizations: 'base64', 'url', or 'none' (default: 'base64') |

**Example Request**:

```bash
curl -X GET \
  https://api.thread-density-analyzer.com/api/results/img_12345678?include_visualizations=true \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

**Response**: Same as `/predict` endpoint response

**Status Codes**:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 401 | Unauthorized - Invalid or missing token |
| 404 | Not Found - Image ID or results not found |
| 500 | Server Error |

### 5. Batch Upload

Upload multiple fabric images for batch processing.

**Endpoint**: `POST /batch/upload`

**Authentication**: Required

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| images[] | File Array | Yes | Array of fabric image files |
| options | JSON String | No | Processing options as JSON string |

**Example Request**:

```bash
curl -X POST \
  https://api.thread-density-analyzer.com/api/batch/upload \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE' \
  -F 'images[]=@fabric1.jpg' \
  -F 'images[]=@fabric2.jpg' \
  -F 'options={"dpi": 300}'
```

**Response**:

```json
{
  "success": true,
  "batch_id": "batch_12345",
  "image_count": 2,
  "images": [
    {
      "image_id": "img_12345678",
      "filename": "fabric1.jpg"
    },
    {
      "image_id": "img_12345679",
      "filename": "fabric2.jpg"
    }
  ],
  "message": "Batch uploaded successfully. Use batch_id to check status."
}
```

**Status Codes**:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid images or parameters |
| 401 | Unauthorized - Invalid or missing token |
| 413 | Payload Too Large - Batch exceeds size limit |
| 500 | Server Error |

### 6. Get Batch Status

Check the status of a batch processing request.

**Endpoint**: `GET /batch/status/{batch_id}`

**Authentication**: Required

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| batch_id | String | Yes | ID of the batch |

**Response**:

```json
{
  "batch_id": "batch_12345",
  "status": "processing",
  "progress": 50,
  "image_count": 2,
  "completed_count": 1,
  "failed_count": 0,
  "images": [
    {
      "image_id": "img_12345678",
      "status": "completed"
    },
    {
      "image_id": "img_12345679",
      "status": "processing"
    }
  ]
}
```

## Error Responses

All API errors return a consistent error format:

```json
{
  "success": false,
  "error": {
    "code": "IMAGE_PROCESSING_ERROR",
    "message": "Unable to process image due to insufficient contrast",
    "details": {
      "suggestion": "Try uploading an image with better lighting conditions"
    }
  }
}
```

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `AUTHENTICATION_ERROR` | Invalid or missing authentication |
| `INVALID_IMAGE` | Image format not supported or corrupted |
| `PROCESSING_ERROR` | General error during image processing |
| `THREAD_DETECTION_FAILED` | Could not detect threads reliably |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `INVALID_PARAMETERS` | Invalid request parameters |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Free tier**: 50 requests per day
- **Basic tier**: 500 requests per day
- **Premium tier**: 5,000 requests per day
- **Enterprise tier**: Custom limits

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 500
X-RateLimit-Remaining: 486
X-RateLimit-Reset: 1622318400
```

## Webhook Notifications

For long-running processes, you can register a webhook to be notified when processing completes:

**Endpoint**: `POST /webhooks/register`

**Request**:

```json
{
  "callback_url": "https://your-app.com/api/thread-analysis-callback",
  "events": ["analysis.completed", "analysis.failed"]
}
```

**Response**:

```json
{
  "webhook_id": "whk_12345",
  "status": "active",
  "events": ["analysis.completed", "analysis.failed"]
}
```

## SDK Availability

We provide client SDKs for easier integration:

- JavaScript/TypeScript: `npm install thread-density-api-client`
- Python: `pip install thread-density-api-client`
- Java: Available on Maven Central

## API Versioning

The API version is included in the URL path:

```
https://api.thread-density-analyzer.com/api/v1/upload
```

The current stable version is `v1`. Deprecated versions will receive a 6-month sunset period before being decommissioned.

---

For questions or support regarding the API, contact us at api-support@thread-density-analyzer.com or open an issue on GitHub.
