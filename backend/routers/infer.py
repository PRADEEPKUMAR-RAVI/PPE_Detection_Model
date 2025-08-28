"""
FastAPI inference router for PPE compliance detection.
"""
import os
import tempfile
import traceback
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import Response, StreamingResponse, FileResponse
import cv2
import json

from backend.services.yolo_service import yolo_service


router = APIRouter(prefix="/infer", tags=["inference"])


@router.post("/image")
async def infer_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0),
    save_outputs: bool = Query(False),
    output_dir: str = Query("outputs")
):
    """
    Run PPE compliance inference on uploaded image.
    Returns annotated image and JSON results.
    """
    try:
        # Debug: Print file information
        print(f"DEBUG - File info: filename={file.filename}, content_type={file.content_type}, size={file.size}")
        
        # Read image bytes first
        image_bytes = await file.read()
        
        # Validate by trying to open the image with PIL (most reliable method)
        try:
            from PIL import Image
            import io
            test_image = Image.open(io.BytesIO(image_bytes))
            test_image.verify()  # Verify it's a valid image
            print(f"DEBUG - Image validated: format={test_image.format}, size={test_image.size}")
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(img_error)}")
        
        # Run inference
        annotated_bytes, result_json = yolo_service.infer_image(
            image_bytes=image_bytes,
            conf_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            save_outputs=save_outputs,
            output_dir=output_dir
        )
        
        # Return JSON response with image data
        response_data = result_json.copy()
        response_data["annotated_image"] = "base64_encoded_image_data_would_be_here"
        
        # For now, return the annotated image directly
        return Response(
            content=annotated_bytes,
            media_type="image/jpeg",
            headers={"X-Result-JSON": json.dumps(result_json)}
        )
        
    except Exception as e:
        print(f"Error in image inference: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video")
async def infer_video(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0),
    save_outputs: bool = Query(True),
    output_dir: str = Query("outputs"),
    download_video: bool = Query(False, description="Return processed video file for download")
):
    """
    Run PPE compliance inference on uploaded video.
    Returns JSON with output video path and results.
    """
    try:
        # Debug: Print file information
        print(f"DEBUG - Video file info: filename={file.filename}, content_type={file.content_type}, size={file.size}")
        
        # Read video bytes first
        video_bytes = await file.read()
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_bytes)
            temp_video_path = temp_file.name
        
        try:
            # Run inference
            output_video_path, result_json = yolo_service.infer_video(
                video_path=temp_video_path,
                conf_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                save_outputs=save_outputs,
                output_dir=output_dir
            )
            
            # Return video file for download if requested
            if download_video and output_video_path and os.path.exists(output_video_path):
                return FileResponse(
                    path=output_video_path,
                    media_type="video/mp4",
                    filename="ppe_detection_result.mp4",
                    headers={"X-Result-JSON": json.dumps(result_json)}
                )
            else:
                return result_json
            
        finally:
            # Clean up temporary input file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/download/{filename}")
async def download_processed_video(filename: str, output_dir: str = Query("outputs")):
    """
    Download a processed video file by filename.
    """
    try:
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Verify it's a video file
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video file format")
        
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/webcam")
async def stream_webcam(
    confidence_threshold: float = Query(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0),
    camera_index: int = Query(0)
):
    """
    Stream MJPEG from webcam with real-time PPE compliance detection.
    """
    def generate_frames():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not access webcam")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference on frame
                annotated_frame, result_json = yolo_service.infer_frame(
                    frame=frame,
                    conf_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in MJPEG format
                result_header = f'X-Result-JSON: {json.dumps(result_json)}\r\n'.encode('utf-8')
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       + result_header +
                       b'\r\n' + frame_bytes + b'\r\n')
        
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/frame")
async def infer_single_frame(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0)
):
    """
    Run inference on a single frame (alternative to streaming for frontend polling).
    """
    try:
        # Debug: Print file information  
        print(f"DEBUG - Frame file info: filename={file.filename}, content_type={file.content_type}, size={file.size}")
        
        # Read image bytes first
        image_bytes = await file.read()
        
        # Validate by trying to open the image with PIL
        try:
            from PIL import Image
            import io
            test_image = Image.open(io.BytesIO(image_bytes))
            test_image.verify()
            print(f"DEBUG - Frame validated: format={test_image.format}, size={test_image.size}")
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(img_error)}")
        
        # Use the same logic as image inference but return JSON only
        annotated_bytes, result_json = yolo_service.infer_image(
            image_bytes=image_bytes,
            conf_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            save_outputs=False,
            output_dir="outputs"
        )
        
        # Return JSON results only
        return result_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        return {
            "model_path": yolo_service.model_path,
            "device": yolo_service.device,
            "class_names": yolo_service.class_names,
            "target_classes": yolo_service.target_classes,
            "model_loaded": yolo_service.model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/reload")
async def reload_model(model_path: Optional[str] = None):
    """Reload the YOLO model with optionally new path."""
    try:
        yolo_service.load_model(model_path)
        return {"message": "Model reloaded successfully", "model_path": yolo_service.model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))