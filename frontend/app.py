"""
Streamlit frontend for PPE Compliance Detection.
"""
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import time
import pandas as pd
import os


# Configuration
API_BASE_URL = "http://localhost:8000"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PPE Compliance Detection",
        page_icon="🦺",
        layout="wide"
    )
    
    st.title("🦺 PPE Compliance Detection")
    st.markdown("YOLOv8-based safety compliance monitoring for Person, Helmet, and Vest detection")
    
    # Sidebar for global settings
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        # Model configuration
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
        save_outputs = st.checkbox("Save Outputs", value=False)
        output_dir = st.text_input("Output Directory", value="outputs")
        
        # Model info
        st.header("📊 Model Info")
        if st.button("Get Model Info"):
            try:
                response = requests.get(f"{API_BASE_URL}/infer/model/info")
                if response.status_code == 200:
                    model_info = response.json()
                    st.json(model_info)
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
        
        # Health check
        st.header("🏥 Health Check")
        if st.button("Check API Health"):
            try:
                response = requests.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    health = response.json()
                    if health.get("status") == "healthy":
                        st.success("API is healthy ✅")
                        st.json(health)
                    else:
                        st.warning("API is unhealthy ⚠️")
                        st.json(health)
                else:
                    st.error("API is down ❌")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Inference", "🎬 Video Inference", "📹 Live Webcam"])
    
    with tab1:
        handle_image_inference(confidence_threshold, iou_threshold, save_outputs, output_dir)
    
    with tab2:
        handle_video_inference(confidence_threshold, iou_threshold, save_outputs, output_dir)
    
    with tab3:
        handle_webcam_inference(confidence_threshold, iou_threshold)


def handle_image_inference(conf_threshold, iou_threshold, save_outputs, output_dir):
    """Handle image inference tab."""
    st.header("Upload Image for PPE Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        key="image_upload"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Run inference button
        if st.button("🔍 Run PPE Detection", key="image_inference"):
            with st.spinner("Running inference..."):
                try:
                    # Prepare API request
                    files = {"file": uploaded_file.getvalue()}
                    params = {
                        "confidence_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                        "save_outputs": save_outputs,
                        "output_dir": output_dir
                    }
                    
                    # Make API call
                    response = requests.post(
                        f"{API_BASE_URL}/infer/image",
                        files={"file": uploaded_file.getvalue()},
                        params=params
                    )
                    
                    if response.status_code == 200:
                        # Get result JSON from headers
                        result_json = json.loads(response.headers.get('X-Result-JSON', '{}'))
                        
                        with col2:
                            st.subheader("Annotated Result")
                            # Display annotated image
                            annotated_image = Image.open(io.BytesIO(response.content))
                            st.image(annotated_image, caption="PPE Detection Result", use_column_width=True)
                            
                            # Download button for annotated image
                            st.download_button(
                                label="📥 Download Annotated Image",
                                data=response.content,
                                file_name="ppe_detection_result.jpg",
                                mime="image/jpeg"
                            )
                        
                        # Display results
                        display_results(result_json)
                        
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")


def handle_video_inference(conf_threshold, iou_threshold, save_outputs, output_dir):
    """Handle video inference tab."""
    st.header("Upload Video for PPE Detection")
    
    uploaded_file = st.file_uploader(
        "Choose a video...",
        type=['mp4', 'avi', 'mov'],
        key="video_upload"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.subheader("Original Video")
        st.video(uploaded_file)
        
        # Video processing options
        col1, col2 = st.columns(2)
        with col1:
            process_only = st.button("🔍 Process Video (JSON Only)", key="video_inference_json")
        with col2:
            process_and_download = st.button("🔍 Process & Download Video", key="video_inference_download")
        
        if process_only or process_and_download:
            with st.spinner("Processing video... This may take a while."):
                try:
                    # Prepare API request
                    files = {"file": uploaded_file.getvalue()}
                    params = {
                        "confidence_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                        "save_outputs": True,  # Always save for video
                        "output_dir": output_dir,
                        "download_video": process_and_download
                    }
                    
                    # Make API call
                    response = requests.post(
                        f"{API_BASE_URL}/infer/video",
                        files={"file": uploaded_file.getvalue()},
                        params=params,
                        timeout=300  # 5 minute timeout for video processing
                    )
                    
                    if response.status_code == 200:
                        if process_and_download:
                            # Handle video file download
                            if 'application/json' in response.headers.get('content-type', ''):
                                # JSON response (no video download)
                                result_json = response.json()
                                display_results(result_json)
                                st.warning("Video processing completed but no video file was returned.")
                            else:
                                # Video file response
                                result_json_header = response.headers.get('X-Result-JSON')
                                if result_json_header:
                                    result_json = json.loads(result_json_header)
                                    display_results(result_json)
                                
                                # Provide video download
                                st.subheader("📥 Processed Video")
                                st.success("Video processing completed!")
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=response.content,
                                    file_name="ppe_detection_result.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            # JSON only response
                            result_json = response.json()
                            display_results(result_json)
                            
                            # Show output video path if available
                            if "output_path" in result_json and result_json["output_path"]:
                                st.subheader("Processed Video")
                                st.success(f"Video saved to: {result_json['output_path']}")
                                
                                # Optional: Try to show download link
                                filename = os.path.basename(result_json["output_path"])
                                download_url = f"{API_BASE_URL}/infer/video/download/{filename}?output_dir={output_dir}"
                                st.markdown(f"🔗 [Download Processed Video]({download_url})")
                        
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")


def handle_webcam_inference(conf_threshold, iou_threshold):
    """Handle webcam inference tab."""
    st.header("Live Webcam PPE Detection")
    
    camera_index = st.number_input("Camera Index", value=0, min_value=0, max_value=10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_webcam = st.button("📹 Start Webcam Stream", key="start_webcam")
        stop_webcam = st.button("⏹️ Stop Stream", key="stop_webcam")
    
    with col2:
        refresh_rate = st.slider("Refresh Rate (seconds)", 0.1, 2.0, 0.5, 0.1)
    
    # Webcam streaming placeholder
    webcam_placeholder = st.empty()
    results_placeholder = st.empty()
    
    # Session state for webcam control
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    if start_webcam:
        st.session_state.webcam_active = True
    
    if stop_webcam:
        st.session_state.webcam_active = False
    
    # Webcam streaming loop
    if st.session_state.webcam_active:
        st.info("🔴 Webcam stream is active. Note: This is a simplified version. For full MJPEG streaming, access the API directly.")
        
        # Alternative: Show webcam stream info
        stream_url = f"{API_BASE_URL}/infer/stream/webcam?confidence_threshold={conf_threshold}&iou_threshold={iou_threshold}&camera_index={camera_index}"
        
        st.markdown(f"""
        **MJPEG Stream URL:**
        ```
        {stream_url}
        ```
        
        **Instructions:**
        1. Open the above URL in a new browser tab to view the live stream
        2. Or use VLC/other media player to open the network stream
        3. The stream includes real-time PPE detection annotations
        """)
        
        # Optionally, try to capture a frame for preview
        if st.button("📸 Capture Frame", key="capture_frame"):
            try:
                # This would require capturing from local webcam
                # For demo purposes, show placeholder
                st.info("Frame capture would show here. Use the stream URL above for full functionality.")
            except Exception as e:
                st.error(f"Error capturing frame: {e}")


def display_results(result_json):
    """Display inference results in a formatted way."""
    st.header("📊 Detection Results")
    
    # Summary counts
    counts = result_json.get("counts", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total People", counts.get("total", 0))
    with col2:
        st.metric("Safe", counts.get("safe", 0), delta=None)
    with col3:
        st.metric("Unsafe", counts.get("unsafe", 0), delta=None)
    
    # Safety percentage
    total = counts.get("total", 0)
    if total > 0:
        safe_percentage = (counts.get("safe", 0) / total) * 100
        st.progress(safe_percentage / 100)
        st.write(f"Safety Compliance: {safe_percentage:.1f}%")
    
    # Detailed person information
    people = result_json.get("people", [])
    if people:
        st.subheader("👥 Individual Detection Details")
        
        # Create DataFrame for better display
        person_data = []
        for i, person in enumerate(people):
            # Check if this is tracked person data (video) or regular detection (image)
            if 'tracking_stats' in person:
                # Video tracking data
                tracking = person['tracking_stats']
                person_data.append({
                    "Person": f"Person {person.get('person_id', i+1)}",
                    "Status": person.get("status", "Unknown"),
                    "Confidence": f"{person.get('confidence', 0):.3f}",
                    "Has Helmet": "✅" if person.get("has_helmet", False) else "❌",
                    "Has Vest": "✅" if person.get("has_vest", False) else "❌",
                    "Appearances": f"{tracking['total_appearances']}",
                    "Safety %": f"{tracking['safety_percentage']}%",
                    "Frames": f"{tracking['first_seen_frame']}-{tracking['last_seen_frame']}"
                })
            else:
                # Regular image detection
                person_data.append({
                    "Person": f"Person {i+1}",
                    "Status": person.get("status", "Unknown"),
                    "Confidence": f"{person.get('confidence', 0):.3f}",
                    "Has Helmet": "✅" if person.get("has_helmet", False) else "❌",
                    "Has Vest": "✅" if person.get("has_vest", False) else "❌",
                    "Bounding Box": f"[{', '.join(map(lambda x: f'{x:.1f}', person.get('bbox', [])))}]"
                })
        
        df = pd.DataFrame(person_data)
        
        # Style the dataframe
        def highlight_status(val):
            color = 'background-color: lightgreen' if val == 'Safe' else 'background-color: lightcoral'
            return color
        
        styled_df = df.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    
    # Video-specific information
    if "video_info" in result_json:
        st.subheader("🎬 Video Information")
        video_info = result_json["video_info"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", video_info.get("total_frames", 0))
        with col2:
            st.metric("Processed Frames", video_info.get("processed_frames", 0))
        with col3:
            st.metric("FPS", video_info.get("fps", 0))
        with col4:
            duration = video_info.get("duration_seconds", 0)
            st.metric("Duration", f"{duration:.1f}s")
        
        # Enhanced analytics
        if "analytics" in result_json:
            st.subheader("📊 Video Analytics")
            analytics = result_json["analytics"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique People", analytics.get("unique_people_detected", 0))
                st.metric("Total Detections", analytics.get("total_person_detections", 0))
                st.metric("Avg People/Frame", analytics.get("average_people_per_frame", 0))
            with col2:
                safety_rate = analytics.get("safety_compliance_rate", 0)
                st.metric("Safety Compliance", f"{safety_rate}%", 
                         delta=f"{'Good' if safety_rate > 80 else 'Needs Improvement'}")
                st.metric("Frames with Detections", analytics.get("frames_with_detections", 0))
                st.metric("Max People in Frame", analytics.get("max_people_in_frame", 0))
            with col3:
                efficiency = analytics.get("processing_efficiency", 0)
                st.metric("Processing Efficiency", f"{efficiency}%")
                tracking_threshold = analytics.get("tracking_threshold", 0)
                st.metric("Tracking Threshold", f"{tracking_threshold} frames")
                st.metric("Resolution", video_info.get("resolution", "Unknown"))


if __name__ == "__main__":
    main()