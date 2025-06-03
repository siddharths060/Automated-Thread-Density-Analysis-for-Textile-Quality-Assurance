#!/bin/bash
# Run backend tests

echo "Running backend unit tests with pytest..."
cd "$(dirname "$0")/backend"
python -m pytest tests/ -v
