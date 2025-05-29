# 🧵 Automated Thread Density Analysis for Textile Quality Assurance

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-v3.8+-blue)
![React](https://img.shields.io/badge/React-v18.0+-61DAFB?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.95.0+-009688?logo=fastapi&logoColor=white)

An end-to-end solution for automated thread counting in textile fabrics to determine quality metrics using computer vision and deep learning.

## 🎯 Problem Statement

Thread count is a critical quality metric in textile manufacturing that traditionally requires manual counting - a tedious, error-prone process. This project automates thread density analysis using deep learning, providing:

- Accurate thread counting for quality assurance
- Consistent evaluation across fabric samples
- Significant time savings compared to manual methods
- Objective measurement without human bias

## 🧠 How Thread Count Works

Thread count is the sum of warp (vertical) and weft (horizontal) threads per square inch of fabric:

```
Thread Count = Warp Thread Count + Weft Thread Count
```

Higher thread counts often indicate higher quality fabrics, especially in bedsheets, clothing, and technical textiles.

## 🔍 U-Net Model for Thread Detection

This project implements an enhanced U-Net architecture with a ResNet backbone for semantic segmentation of fabric images. The U-Net model was chosen for its ability to:

- Preserve fine details through skip connections
- Segment thin structures (individual threads)
- Maintain context while detecting local patterns
- Perform well with limited training data

Our implementation combines advanced preprocessing techniques with deep learning to achieve high accuracy in thread detection.

## ⚙️ Technologies Used

### Frontend
- **React.js**: Interactive UI for image uploading and result visualization
- **Material UI**: Modern component library for responsive design
- **Chart.js**: Visualization of thread count metrics
- **Axios**: API communication with backend

### Backend
- **FastAPI**: High-performance Python web framework
- **PyTorch**: Deep learning framework for model inference
- **OpenCV**: Computer vision library for image processing
- **NumPy/SciPy**: Scientific computing for numerical operations

## 🚀 How It Works

![System Architecture](docs/assets/architecture-diagram.png)

1. User uploads a fabric image through the React frontend
2. Backend preprocesses the image (normalization, enhancement)
3. U-Net model generates a segmentation mask highlighting threads
4. Post-processing algorithms separate and count warp/weft threads
5. Results are returned to the frontend for visualization
6. Thread count metrics and annotated images are displayed to the user

## 💻 Setup Instructions

### Prerequisites
- Node.js v14+
- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+

### Frontend Setup
```bash
# Clone the repository
git clone https://github.com/siddharthss/automated-thread-density-analysis.git
cd automated-thread-density-analysis

# Install frontend dependencies
cd frontend
npm install
npm start
```

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```

### Model Setup
```bash
# Download pre-trained model weights
python scripts/download_model.py

# Verify model setup
python scripts/test_model.py
```

## 📊 Sample Results

| Input Image | Thread Detection | Results |
|-------------|------------------|---------|
| ![Sample Fabric 1](docs/assets/sample-input-1.jpg) | ![Detected Threads 1](docs/assets/thread-detection-1.jpg) | Warp: 180<br>Weft: 120<br>Total: 300 |
| ![Sample Fabric 2](docs/assets/sample-input-2.jpg) | ![Detected Threads 2](docs/assets/thread-detection-2.jpg) | Warp: 148<br>Weft: 92<br>Total: 240 |

## 📈 Future Scope

- **Real-time Analysis**: Integration with camera feeds for on-the-fly inspection
- **Mobile App**: Development of mobile applications for field use
- **Quality Classification**: Automatic grading of fabrics (low/medium/high quality)
- **Defect Detection**: Expanding the model to identify fabric defects
- **Material Type Recognition**: Automatic identification of fabric type
- **Industry 4.0 Integration**: API for seamless integration with manufacturing systems

## 🗂️ Project Structure

```
automated-thread-density-analysis/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── utils/
│   │   └── App.js
│   ├── package.json
│   └── README.md
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   ├── models/
│   │   └── services/
│   ├── ml/
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   └── thread_counter.py
│   ├── tests/
│   ├── main.py
│   └── requirements.txt
├── model/
│   ├── weights/
│   ├── training/
│   └── evaluation/
└── docs/
    ├── assets/
    ├── model-details.md
    ├── thread-counting.md
    ├── api-spec.md
    └── examples/
```

## 🤝 Contributing

We welcome contributions to improve the Automated Thread Density Analysis project! Please see our [contribution guidelines](CONTRIBUTING.md) for details on how to participate.

## 🪪 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Useful Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React.js Documentation](https://reactjs.org/docs/getting-started.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Textile Industry Standards for Thread Count](https://www.astm.org/d3775-17r21.html)

---

Developed with ❤️ by Siddharth S
