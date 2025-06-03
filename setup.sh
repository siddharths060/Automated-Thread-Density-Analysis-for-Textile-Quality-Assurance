#!/bin/bash

# Setup script for Automated Thread Density Analysis

echo "🧵 Setting up Automated Thread Density Analysis Tool..."
echo "----------------------------------------------------"

# Navigate to project root
PROJECT_ROOT=$(pwd)
echo "📁 Project root: $PROJECT_ROOT"

# Setup Backend
echo -e "\n📦 Setting up backend..."
cd "$PROJECT_ROOT/backend"

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
  echo "🔧 Creating Python virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

# Setup Frontend
echo -e "\n📦 Setting up frontend..."
cd "$PROJECT_ROOT/frontend"

# Install dependencies
echo "🔧 Installing Node.js dependencies..."
npm install

echo -e "\n✅ Setup complete!"
echo "----------------------------------------------------"
echo "To run the backend: cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo "To run the frontend: cd frontend && npm start"
echo "----------------------------------------------------"
