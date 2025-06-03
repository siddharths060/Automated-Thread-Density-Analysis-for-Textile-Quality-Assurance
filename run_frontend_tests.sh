#!/bin/bash
# Run frontend tests

echo "Running frontend unit tests..."
cd "$(dirname "$0")/frontend"
npm test -- --watchAll=false
