#!/bin/bash

# PPE Compliance Detection System Launch Script
# This script starts both the FastAPI backend and Streamlit frontend

echo "🦺 Starting PPE Compliance Detection System..."
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if requirements are installed
if [ ! -f "venv/lib/python*/site-packages/ultralytics/__init__.py" ]; then
    echo "❌ Dependencies not installed. Please run:"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if model file exists
if [ ! -f "backend/models/best.pt" ]; then
    echo "❌ Model file not found at backend/models/best.pt"
    echo "   Please copy your trained YOLOv8 model to this location"
    exit 1
fi

# Create outputs directory
mkdir -p outputs

echo "✅ Pre-flight checks passed"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "   Stopped backend (PID: $BACKEND_PID)"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "   Stopped frontend (PID: $FRONTEND_PID)"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Activate virtual environment
source venv/bin/activate

echo "🚀 Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

echo "🌐 Starting Streamlit frontend..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "✅ Services started successfully!"
echo "================================="
echo "🔗 FastAPI Backend:  http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🖥️  Streamlit Frontend: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   - Use Ctrl+C to stop both services"
echo "   - Check the logs below for any errors"
echo "   - Ensure your webcam is not used by other apps for live streaming"
echo ""
echo "📊 Service Status:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "🔍 Waiting for services... (Press Ctrl+C to stop)"

# Wait for both processes
wait