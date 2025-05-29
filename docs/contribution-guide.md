# 🤝 Contribution Guide

Thank you for your interest in contributing to the Automated Thread Density Analysis project! This guide provides information on how to contribute to the project effectively.

## 📋 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive of all contributors
- Provide constructive feedback
- Focus on the improvement of the project
- Be patient with new contributors
- Refrain from any offensive or disrespectful behavior

## 🛠️ Setting Up the Development Environment

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Node.js 14+
- Git
- Virtual environment tool (like venv or conda)
- A good code editor (VS Code recommended)

### Fork and Clone the Repository

1. Fork the repository by clicking the "Fork" button at the top right of the [repository page](https://github.com/yourusername/thread-density-analysis).

2. Clone your fork to your local machine:

```bash
git clone https://github.com/your-username/thread-density-analysis.git
cd thread-density-analysis
```

3. Add the original repository as an upstream remote:

```bash
git remote add upstream https://github.com/original-owner/thread-density-analysis.git
```

### Backend Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the development server:

```bash
uvicorn main:app --reload
```

### Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the development server:

```bash
npm start
```

## 🌲 Project Structure

```
thread-density-analysis/
│
├── backend/                       # FastAPI backend
│   ├── api/                       # API endpoints
│   ├── core/                      # Core business logic
│   ├── ml/                        # Machine learning models
│   ├── utils/                     # Utility functions
│   └── main.py                    # Application entry point
│
├── frontend/                      # React.js frontend
│   ├── public/                    # Static assets
│   ├── src/                       # Source code
│   │   ├── components/            # React components
│   │   ├── hooks/                 # Custom React hooks
│   │   ├── services/              # API service calls
│   │   └── utils/                 # Utility functions
│   └── package.json               # NPM configuration
│
├── docs/                          # Documentation
│   ├── api-spec.md                # API specification
│   ├── architecture.md            # System architecture
│   └── ...                        # Other documentation
│
├── tests/                         # Test suite
│   ├── backend/                   # Backend tests
│   └── frontend/                  # Frontend tests
│
├── scripts/                       # Utility scripts
│   └── setup.sh                   # Setup script
│
├── .github/                       # GitHub workflows
│   └── workflows/                 # CI/CD workflow definitions
│
├── README.md                      # Project overview
└── LICENSE                        # Project license
```

## 🌿 Branching Strategy

We follow a simplified Git flow:

- `main`: Production-ready code
- `develop`: Integration branch for new features
- `feature/*`: New features or enhancements
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent fixes for production
- `release/*`: Release preparation

### Creating a New Feature Branch

```bash
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name
```

## 📝 Making Changes

### Coding Standards

- Follow PEP 8 for Python code
- Use ESLint rules for JavaScript/TypeScript code
- Write clear, self-documenting code
- Add comments for complex logic
- Include docstrings for functions and classes

### Documentation

- Update documentation when changing functionality
- Document any new features or APIs
- Include docstrings for all public functions, classes, and methods
- Comment complex code sections

### Testing

- Write tests for new features
- Ensure existing tests pass
- Backend: Use pytest for unit and integration tests
- Frontend: Use Jest and React Testing Library

## 📦 Committing Changes

### Commit Messages

Follow the conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat(thread-detection): add support for diagonal threads

- Implement Hough transform for diagonal line detection
- Add parameter for diagonal angle tolerance
- Update visualization to highlight diagonal threads

Closes #42
```

### Pull Requests

1. Push your changes to your fork:

```bash
git push origin feature/your-feature-name
```

2. Go to the original repository and create a Pull Request:
   - Base: `develop`
   - Compare: `feature/your-feature-name`

3. Fill in the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots if applicable

4. Wait for CI checks to complete
5. Respond to review feedback

## 🚀 Adding a New Model

To add a new thread detection model:

1. Create a new class in `backend/ml/models/`:

```python
class YourThreadModel(BaseThreadModel):
    def __init__(self, params):
        super().__init__()
        # Initialize your model
    
    def preprocess(self, image):
        # Preprocess image for your model
        return processed_image
    
    def predict(self, image):
        # Run inference with your model
        return prediction
    
    def postprocess(self, prediction):
        # Process raw model output
        return thread_mask
```

2. Register your model in `backend/ml/model_factory.py`
3. Add model-specific parameters in `backend/models/request.py`
4. Write tests in `tests/backend/ml/test_your_model.py`
5. Update documentation in `docs/model-details.md`

## 📊 Writing Test Cases

### Backend Tests

```python
# tests/backend/test_thread_counter.py
import pytest
from backend.core.thread_counter import ThreadCounter

def test_thread_counting_accuracy():
    """Test thread counting accuracy on a known sample."""
    # Arrange
    counter = ThreadCounter()
    known_sample = load_test_image("cotton_200tc.jpg")
    expected_warp = 120
    expected_weft = 80
    
    # Act
    warp, weft, _, _ = counter.count_threads(known_sample)
    
    # Assert
    assert abs(warp - expected_warp) <= 5  # Allow 5 thread margin of error
    assert abs(weft - expected_weft) <= 5
```

### Frontend Tests

```javascript
// tests/frontend/components/ThreadCountDisplay.test.js
import { render, screen } from '@testing-library/react';
import ThreadCountDisplay from '../../src/components/ThreadCountDisplay';

test('displays thread count correctly', () => {
  // Arrange
  const threadData = {
    warp: 120,
    weft: 80,
    total: 200
  };
  
  // Act
  render(<ThreadCountDisplay threadData={threadData} />);
  
  // Assert
  expect(screen.getByText(/120/i)).toBeInTheDocument(); // Warp count
  expect(screen.getByText(/80/i)).toBeInTheDocument();  // Weft count
  expect(screen.getByText(/200/i)).toBeInTheDocument(); // Total count
});
```

## 📋 Opening Issues

### Bug Reports

When filing a bug report, please include:

1. Clear and descriptive title
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Screenshots if applicable
6. Environment details (OS, browser, etc.)
7. Possible solution (if you have one)

### Feature Requests

When requesting a new feature:

1. Describe the problem your feature would solve
2. Outline the proposed solution
3. Discuss alternatives you've considered
4. Explain how this feature would benefit the project
5. Include mockups or examples if possible

## 📚 Style Guide

### Python Style Guide

- Follow PEP 8
- Maximum line length: 88 characters (Black formatter)
- Use type hints when possible
- Sort imports using isort
- Format code with Black

Example:
```python
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .utils import normalize_image


def process_fabric_image(
    image_path: str, 
    dpi: Optional[float] = None
) -> Dict[str, any]:
    """
    Process a fabric image to detect and count threads.
    
    Args:
        image_path: Path to the fabric image file
        dpi: Optional DPI override for the image
        
    Returns:
        Dictionary with thread count results
    """
    # Implementation
    pass
```

### JavaScript/TypeScript Style Guide

- Use ESLint and Prettier
- Follow Airbnb style guide
- Use functional components with hooks for React
- Prefer TypeScript over JavaScript

Example:
```tsx
import React, { useState, useEffect } from 'react';
import { ThreadCountResult } from '../types';
import { fetchThreadCount } from '../services/api';

interface ThreadCounterProps {
  imageId: string;
  onComplete?: (result: ThreadCountResult) => void;
}

const ThreadCounter: React.FC<ThreadCounterProps> = ({ imageId, onComplete }) => {
  const [result, setResult] = useState<ThreadCountResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const loadResults = async (): Promise<void> => {
      try {
        setLoading(true);
        const data = await fetchThreadCount(imageId);
        setResult(data);
        if (onComplete) {
          onComplete(data);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    loadResults();
  }, [imageId, onComplete]);
  
  // Component rendering
};

export default ThreadCounter;
```

## 🙌 Recognition

Contributors will be recognized in the following ways:

- Listed in the Contributors section of the README
- Mentioned in release notes for significant contributions
- Given credit in documentation for specific features

## 💬 Getting Help

If you need help with your contribution:

- Open a discussion on GitHub
- Ask in the issue you're working on
- Contact the maintainers via email

We appreciate all contributions, from code to documentation to bug reports!

## 🔄 Keeping Your Fork Updated

To keep your fork updated with the main repository:

```bash
git fetch upstream
git checkout develop
git merge upstream/develop
git push origin develop
```

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](../LICENSE).
