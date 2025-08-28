"""
FastAPI main application for PPE compliance detection.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.routers import infer


# Create FastAPI app
app = FastAPI(
    title="PPE Compliance Detection API",
    description="YOLOv8-based API for detecting PPE compliance in images and videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(infer.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PPE Compliance Detection API",
        "version": "1.0.0",
        "endpoints": {
            "image_inference": "/infer/image",
            "video_inference": "/infer/video",
            "webcam_stream": "/infer/stream/webcam",
            "frame_inference": "/infer/frame",
            "model_info": "/infer/model/info",
            "model_reload": "/infer/model/reload"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from backend.services.yolo_service import yolo_service
        model_loaded = yolo_service.model is not None
        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "device": yolo_service.device
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )