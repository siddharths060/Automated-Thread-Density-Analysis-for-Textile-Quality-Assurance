# 🤝 Contributing to Automated Thread Density Analysis

Thank you for your interest in contributing to the Automated Thread Density Analysis project! This document provides guidelines and instructions for contributing effectively.

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be Respectful**: Treat everyone with respect regardless of experience level, gender, gender identity, sexual orientation, disability, race, ethnicity, age, religion, or nationality.
- **Be Constructive**: Provide constructive feedback and criticism. Help improve the project, not discourage participation.
- **Be Collaborative**: Work together to find solutions. Focus on what is best for the project community as a whole.
- **Be Professional**: Disagreements happen, but maintain professional conduct. Harassment or disrespectful behavior will not be tolerated.

## 🚀 Getting Started

### Fork and Clone the Repository

1. Fork the repository on GitHub by clicking the "Fork" button
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR-USERNAME/automated-thread-density-analysis.git
cd automated-thread-density-analysis
```

3. Add the original repository as an upstream remote:
```bash
git remote add upstream https://github.com/siddharthss/automated-thread-density-analysis.git
```

### Setting Up Development Environment

#### Frontend (React.js)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

#### Backend (FastAPI + Python)

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload
```

#### ML Model Development

```bash
# Install additional dependencies for model development
pip install -r model/requirements.txt

# Run model tests
pytest model/tests/
```

## 🌿 Branching Conventions

We follow a simplified Git branching model:

- `main`: The production-ready branch
- `development`: The primary development branch
- `feature/*`: For new features (e.g., `feature/mobile-support`)
- `bugfix/*`: For bug fixes (e.g., `bugfix/preprocessing-error`)
- `hotfix/*`: For critical production fixes (e.g., `hotfix/security-issue`)

### Workflow

1. Create a new branch from `development` for your work:
```bash
git checkout development
git pull upstream development
git checkout -b feature/your-feature-name
```

2. Make your changes, commit, and push to your fork:
```bash
git add .
git commit -m "Add feature: your feature description"
git push origin feature/your-feature-name
```

3. Open a pull request against the `development` branch

## 📝 Pull Request Process

1. Ensure your code follows our style guidelines
2. Include relevant tests for your changes
3. Update documentation if necessary
4. Fill out the pull request template completely
5. Request a review from one or more maintainers
6. Be responsive to feedback and make requested changes
7. Once approved, a maintainer will merge your PR

### Pull Request Template

When creating a pull request, please include:

- **Purpose**: What does this PR do?
- **Related Issue**: Link to the issue this PR addresses (if applicable)
- **Changes Made**: Summary of key changes
- **Testing**: How were the changes tested?
- **Screenshots**: (if applicable, especially for UI changes)
- **Additional Notes**: Any information that might be helpful for reviewers

## 🧪 Testing Guidelines

All contributions should include appropriate tests:

### Frontend Testing

- Unit tests for components using Jest and React Testing Library
- Integration tests for key user flows
- Maintain >80% test coverage for new code

```bash
# Run frontend tests
cd frontend
npm test
```

### Backend Testing

- Unit tests for API endpoints and services
- Integration tests for data flow
- Test edge cases and error handling

```bash
# Run backend tests
cd backend
pytest
```

### Model Testing

- Unit tests for preprocessing functions
- Integration tests for the model pipeline
- Validation tests with benchmark datasets

```bash
# Run model tests
cd model
pytest tests/
```

## 📚 Documentation Standards

Good documentation is essential for this project:

### Code Documentation

- Use descriptive variable and function names
- Add comments for complex logic
- Include docstrings for all functions, classes, and methods:

```python
def calculate_thread_count(mask, orientation='warp'):
    """
    Calculate thread count from a binary segmentation mask.
    
    Args:
        mask (numpy.ndarray): Binary segmentation mask
        orientation (str): Thread orientation, either 'warp' or 'weft'
        
    Returns:
        int: Number of threads detected
    
    Raises:
        ValueError: If orientation is not 'warp' or 'weft'
    """
    # Implementation...
```

### API Documentation

- Document all API endpoints using FastAPI's built-in tools
- Include example requests and responses

### User Documentation

- Keep the README.md up-to-date
- Update relevant documentation files in the `docs/` directory

## 💻 Coding Standards

### Python Code

- Follow PEP 8 style guide
- Use type hints
- Max line length: 100 characters
- Use meaningful variable names

```python
# Good
def process_image(image_path: str, resize_dimensions: Tuple[int, int]) -> np.ndarray:
    """Process the input image."""
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, resize_dimensions)
    return resized_image

# Avoid
def proc_img(path, dims):
    img = cv2.imread(path)
    return cv2.resize(img, dims)
```

### JavaScript/React Code

- Use ESLint with our configuration
- Use functional components with hooks when possible
- Follow consistent naming conventions:
  - Components: PascalCase
  - Functions: camelCase
  - Files: kebab-case for components, camelCase for utilities

```javascript
// Good
function ThreadCountDisplay({ warpCount, weftCount }) {
  const totalCount = warpCount + weftCount;
  
  return (
    <div className="thread-count-display">
      <p>Warp: {warpCount}</p>
      <p>Weft: {weftCount}</p>
      <p>Total: {totalCount}</p>
    </div>
  );
}

// Avoid
function thread_count(props) {
  var total = props.warp + props.weft;
  return <div><p>Total: {total}</p></div>;
}
```

## 🚀 Feature Requests and Bugs

### Feature Requests

1. Check if the feature has already been requested or implemented
2. Use the feature request template
3. Be specific about the use case and benefits
4. Consider submitting a PR if you can implement the feature

### Bug Reports

1. Check if the bug has already been reported
2. Use the bug report template
3. Include steps to reproduce the issue
4. Describe expected vs. actual behavior
5. Include environment details (OS, browser, versions)
6. Add screenshots if applicable

## 📋 Project Structure

Understanding our project structure will help you contribute effectively:

```
automated-thread-density-analysis/
├── frontend/                # React.js frontend
│   ├── public/              # Static assets
│   ├── src/                 # Source code
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API services
│   │   └── utils/           # Utility functions
│   └── tests/               # Frontend tests
│
├── backend/                 # FastAPI backend
│   ├── api/                 # API endpoints
│   ├── core/                # Core functionality
│   ├── ml/                  # ML model integration
│   ├── services/            # Business logic
│   └── tests/               # Backend tests
│
├── model/                   # Deep learning model
│   ├── data/                # Training/validation data
│   ├── notebooks/           # Experimentation notebooks
│   ├── src/                 # Model implementation
│   └── tests/               # Model tests
│
└── docs/                    # Documentation
```

## 👨‍💻 Getting Help

If you need help with contributing:

1. Check the documentation in the `docs/` directory
2. Open a question in GitHub Discussions
3. Join our community Slack channel
4. Contact the maintainers directly

## 🙏 Recognition

Contributors will be recognized in:

- The project README.md
- Our contributors page
- Release notes when features are included

Thank you for contributing to the Automated Thread Density Analysis project! Your efforts help improve textile quality assessment for everyone.

---

📅 Last Updated: May 2025
