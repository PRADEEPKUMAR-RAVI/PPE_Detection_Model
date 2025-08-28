# PPE Compliance Detection System

A complete YOLOv8-based PPE (Personal Protective Equipment) compliance detection system with FastAPI backend and Streamlit frontend.

## Features

- **Real-time PPE Detection**: Detects Person, Helmet, and Vest classes
- **Compliance Checking**: Determines safety status based on PPE presence
- **Multiple Input Types**: Support for images, videos, and live webcam streams
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **Streamlit Frontend**: User-friendly web interface
- **CUDA Support**: Automatic GPU acceleration when available

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- Webcam (optional, for live streaming)

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI app & routes
│   ├── routers/
│   │   └── infer.py         # image/video/webcam endpoints
│   ├── services/
│   │   └── yolo_service.py  # model loading + inference logic
│   ├── utils/
│   │   └── geometry.py      # IoU/containment helpers
│   └── models/
│       └── best.pt          # YOLOv8 trained model
├── frontend/
│   └── app.py               # Streamlit UI
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── run.sh                  # Launch script
```

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/PPE_model
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your trained model**:
   - Copy your YOLOv8 model file (`best.pt`) to `backend/models/best.pt`
   - The model should be trained on classes: ['Earplug','Gloves','Goggles','Helmet','Mask','Person','Shoes','Vest']

## Usage

### Method 1: Using the Launch Script

1. **Make the script executable** (Linux/Mac):
   ```bash
   chmod +x run.sh
   ```

2. **Run the application**:
   ```bash
   ./run.sh
   ```

### Method 2: Manual Launch

1. **Start the FastAPI backend** (in terminal 1):
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

2. **Start the Streamlit frontend** (in terminal 2):
   ```bash
   streamlit run frontend/app.py
   ```

## API Endpoints

### Backend API (http://localhost:8000)

- **POST /infer/image**: Upload image for PPE detection
- **POST /infer/video**: Upload video for PPE detection
- **GET /infer/stream/webcam**: MJPEG webcam stream with real-time detection
- **POST /infer/frame**: Single frame inference
- **GET /infer/model/info**: Get model information
- **POST /infer/model/reload**: Reload model with new path
- **GET /health**: API health check
- **GET /docs**: Interactive API documentation

### Parameters

All inference endpoints support these query parameters:

- `confidence_threshold`: Detection confidence threshold (default: 0.25)
- `iou_threshold`: IoU threshold for NMS (default: 0.45)
- `save_outputs`: Save annotated results to disk (default: false)
- `output_dir`: Directory for saved outputs (default: "outputs")

## Frontend Features (http://localhost:8501)

The Streamlit interface provides three main tabs:

### 1. Image Inference
- Upload images (JPG, PNG)
- Real-time PPE detection
- Download annotated results
- Per-person compliance details

### 2. Video Inference
- Upload videos (MP4, AVI, MOV)
- Batch processing with progress tracking
- Comprehensive analytics
- Video output generation

### 3. Live Webcam
- Real-time webcam streaming
- Live PPE compliance monitoring
- MJPEG stream access
- Configurable refresh rates

## PPE Compliance Rules

The system applies the following safety compliance logic:

- **SAFE** (Green box): Person has BOTH Helmet AND Vest detected within their bounding box
- **UNSAFE** (Red box): Person is missing either Helmet, Vest, or both

## Model Configuration

### Default Classes
The system expects a YOLOv8 model trained on 8 classes:
```python
['Earplug', 'Gloves', 'Goggles', 'Helmet', 'Mask', 'Person', 'Shoes', 'Vest']
```

### Target Classes for Compliance
Only these classes are used for compliance checking:
- **Person** (class_id: 5)
- **Helmet** (class_id: 3) 
- **Vest** (class_id: 7)

### Customization
To use a different model or classes:
1. Update the `class_names` and `target_classes` in `backend/services/yolo_service.py`
2. Place your model file in `backend/models/`
3. Use the model reload endpoint or restart the service

## Device Selection

The system automatically selects the best available device:
- **CUDA GPU**: Used if available for faster inference
- **CPU**: Fallback option for systems without CUDA

## Output Format

### JSON Response Schema
```json
{
  "source": "image|video|webcam",
  "people": [
    {
      "bbox": [x1, y1, x2, y2],
      "status": "Safe|Unsafe",
      "has_helmet": true|false,
      "has_vest": true|false,
      "confidence": 0.95
    }
  ],
  "counts": {
    "safe": 2,
    "unsafe": 1,
    "total": 3
  },
  "output_path": "outputs/annotated_result.jpg"
}
```

## Troubleshooting

### Common Issues

1. **Model not found error**:
   - Ensure `best.pt` is in `backend/models/` directory
   - Check file permissions

2. **CUDA out of memory**:
   - Reduce image/video resolution
   - Lower batch size in video processing
   - Use CPU inference instead

3. **Webcam not accessible**:
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure camera isn't used by other applications

4. **API connection errors**:
   - Verify backend is running on port 8000
   - Check firewall settings
   - Ensure virtual environment is activated

### Performance Tips

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster inference
- **Image Size**: Resize large images before processing
- **Confidence Thresholds**: Adjust thresholds based on your use case
- **Video Processing**: Use lower resolution videos for faster processing

## Development

### Adding New Features

1. **Backend**: Add new endpoints in `backend/routers/infer.py`
2. **Frontend**: Extend Streamlit interface in `frontend/app.py`
3. **Processing**: Modify inference logic in `backend/services/yolo_service.py`

### Testing

1. **API Testing**: Use the interactive docs at http://localhost:8000/docs
2. **Health Check**: GET http://localhost:8000/health
3. **Model Info**: GET http://localhost:8000/infer/model/info

## License

This project is for educational and safety monitoring purposes. Ensure compliance with local regulations when deploying for commercial use.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure the model file is properly placed and accessible
4. Check logs in both backend and frontend terminals