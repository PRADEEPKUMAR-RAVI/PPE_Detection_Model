"""
YOLOv8 service for PPE compliance detection with enhanced tracking.
"""
import os
import io
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
from PIL import Image
import torch

from backend.utils.geometry import is_contained, calculate_containment_ratio, calculate_iou
import collections
from collections import deque


class Track:
    """Individual track for a person with enhanced lifecycle management."""
    
    def __init__(self, track_id: int, initial_bbox: List[float], initial_status: Dict):
        self.track_id = track_id
        self.bbox = initial_bbox
        self.confidence = initial_status.get('confidence', 0.0)
        self.status = initial_status.get('status', 'Unknown')
        self.has_helmet = initial_status.get('has_helmet', False)
        self.has_vest = initial_status.get('has_vest', False)
        self.ppe_detected = initial_status.get('ppe_detected', {})
        
        # Track lifecycle state management
        self.state = 'tentative'  # tentative -> confirmed -> lost -> deleted
        self.age = 0  # Total number of frames since track creation
        self.time_since_update = 0  # Frames since last successful match
        self.hits = 1  # Number of successful matches
        self.hit_streak = 1  # Current consecutive successful matches
        
        # Temporal smoothing buffers
        self.bbox_buffer = deque([initial_bbox], maxlen=5)  # Last 5 bounding boxes
        self.confidence_buffer = deque([self.confidence], maxlen=3)
        
        # Statistics tracking
        self.total_appearances = 1
        self.safe_count = 1 if self.status == 'Safe' else 0
        self.unsafe_count = 1 if self.status == 'Unsafe' else 0
        self.first_frame = 0
        self.last_frame = 0
        
        # Standing detection tracking
        self.is_standing = initial_status.get('is_standing', False)
        self.standing_confidence = initial_status.get('standing_confidence', 0.0)
        self.standing_frames = 1 if self.is_standing else 0
        self.total_standing_time = 0.0  # In seconds
        self.current_standing_start = None  # Frame number when current standing session started
        
        # Initialize standing start if person is standing
        if self.is_standing:
            self.current_standing_start = 0  # Start counting from frame 0
        
        # Motion prediction parameters
        self.velocity = [0, 0, 0, 0]  # dx, dy, dw, dh
        self.velocity_buffer = deque(maxlen=3)
    
    def get_quality_score(self) -> float:
        """Calculate track quality based on multiple factors."""
        # Base score from hit rate
        hit_rate = self.hits / max(1, self.age)
        
        # Confidence factor
        avg_conf = sum(self.confidence_buffer) / len(self.confidence_buffer)
        
        # Streak bonus
        streak_factor = min(1.0, self.hit_streak / 5)
        
        # Recency factor (prefer recently updated tracks)
        recency_factor = max(0.1, 1.0 - (self.time_since_update / 10))
        
        return hit_rate * 0.4 + avg_conf * 0.3 + streak_factor * 0.2 + recency_factor * 0.1
    
    def predict(self):
        """Enhanced prediction with momentum-based motion model."""
        try:
            self.age += 1
            self.time_since_update += 1
            
            if len(self.bbox_buffer) >= 2:
                # Momentum-based prediction using multiple frames
                bbox_list = list(self.bbox_buffer)
                
                # Calculate weighted velocity from multiple frames
                if len(bbox_list) >= 3:
                    # Use last 3 frames for smoother prediction
                    velocities = []
                    for i in range(len(bbox_list) - 2, len(bbox_list)):
                        curr = bbox_list[i]
                        prev = bbox_list[i-1]
                        vel = [(curr[j] - prev[j]) for j in range(4)]
                        velocities.append(vel)
                    
                    # Average velocities with more weight on recent
                    weights = [0.3, 0.7] if len(velocities) == 2 else [0.2, 0.3, 0.5][-len(velocities):]
                    avg_velocity = [0, 0, 0, 0]
                    for v, w in zip(velocities, weights):
                        for j in range(4):
                            avg_velocity[j] += v[j] * w
                    
                    # Apply momentum with damping
                    damping = 0.8  # Reduce velocity over time
                    self.velocity = [v * damping for v in avg_velocity]
                else:
                    # Simple velocity from last two frames
                    curr = bbox_list[-1]
                    prev = bbox_list[-2]
                    self.velocity = [(curr[j] - prev[j]) * 0.7 for j in range(4)]
                
                # Apply prediction with bounds checking
                predicted_bbox = [
                    max(0, self.bbox[j] + self.velocity[j]) for j in range(4)
                ]
                
                # Ensure reasonable bounding box dimensions
                if predicted_bbox[2] > 10 and predicted_bbox[3] > 10:  # min width/height
                    self.bbox = predicted_bbox
                    self.bbox_buffer.append(predicted_bbox)
                    
        except Exception as e:
            print(f"Prediction error for track {self.track_id}: {e}")
    
    def update(self, new_bbox: List[float], detection: Dict, frame_number: int, fps: float = 30.0):
        """Update track with new detection."""
        self.bbox = new_bbox
        self.bbox_buffer.append(new_bbox)
        
        # Update confidence with temporal smoothing
        new_conf = detection.get('confidence', 0.0)
        self.confidence_buffer.append(new_conf)
        self.confidence = sum(self.confidence_buffer) / len(self.confidence_buffer)
        
        # Update status and PPE info
        self.status = detection.get('status', 'Unknown')
        self.has_helmet = detection.get('has_helmet', False)
        self.has_vest = detection.get('has_vest', False)
        self.ppe_detected = detection.get('ppe_detected', {})
        
        # Track lifecycle management
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.total_appearances += 1
        self.last_frame = frame_number
        
        # Update safety statistics
        if self.status == 'Safe':
            self.safe_count += 1
        else:
            self.unsafe_count += 1
        
        # Update standing statistics
        new_is_standing = detection.get('is_standing', False)
        new_standing_confidence = detection.get('standing_confidence', 0.0)
        
        if new_is_standing:
            self.standing_frames += 1
            if not self.is_standing:  # Just started standing
                self.current_standing_start = frame_number
        else:
            if self.is_standing and self.current_standing_start is not None:  # Just stopped standing
                # Add to total standing time
                frames_standing = frame_number - self.current_standing_start
                time_standing = frames_standing / fps
                self.total_standing_time += time_standing
                self.current_standing_start = None
        
        self.is_standing = new_is_standing
        self.standing_confidence = new_standing_confidence
        
        # State transitions
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark track as missed this frame."""
        self.hit_streak = 0
        self.time_since_update += 1
        
        # State transitions
        if self.state == 'confirmed' and self.time_since_update > 5:
            self.state = 'lost'
        elif self.state == 'tentative' and self.time_since_update > 3:
            self.state = 'lost'
    
    def should_delete(self) -> bool:
        """Determine if track should be deleted."""
        # Delete lost tracks after extended period
        if self.state == 'lost' and self.time_since_update > 10:
            return True
        
        # Delete very old tentative tracks
        if self.state == 'tentative' and self.age > 30:
            return True
        
        # Delete tracks with very low quality
        if self.get_quality_score() < 0.1 and self.age > 10:
            return True
            
        return False
    
    def get_safety_percentage(self) -> float:
        """Calculate safety percentage for this track."""
        if self.total_appearances == 0:
            return 0.0
        return (self.safe_count / self.total_appearances) * 100
    
    def get_standing_percentage(self) -> float:
        """Calculate standing percentage for this track."""
        if self.total_appearances == 0:
            return 0.0
        return (self.standing_frames / self.total_appearances) * 100
    
    def finalize_standing_time(self, final_frame: int, fps: float = 30.0) -> float:
        """Finalize standing time calculation when track ends."""
        if self.is_standing and self.current_standing_start is not None:
            # Add remaining standing time
            frames_standing = final_frame - self.current_standing_start
            time_standing = frames_standing / fps
            self.total_standing_time += time_standing
            self.current_standing_start = None
        return self.total_standing_time
    
    def to_dict(self, interpolated: bool = False) -> Dict:
        """Convert track to dictionary format."""
        return {
            'person_id': self.track_id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'status': self.status,
            'has_helmet': self.has_helmet,
            'has_vest': self.has_vest,
            'ppe_detected': self.ppe_detected,  # This was missing!
            'interpolated': interpolated,
            'quality_score': self.get_quality_score(),
            'total_appearances': self.total_appearances,
            'is_standing': self.is_standing,
            'standing_confidence': self.standing_confidence,
            'total_standing_time': self.total_standing_time,
            'standing_percentage': self.get_standing_percentage()
        }


class MultiObjectTracker:
    """Enhanced multi-object tracker with ByteTrack-style lifecycle management."""
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 150, fps: float = 30.0):
        self.tracks = {}  # Dict of track_id -> Track
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.fps = fps  # Frames per second for time calculations
        
        # Dynamic thresholds (lowered for better low-quality tracking)
        self.high_iou_threshold = 0.25  # For confirmed tracks
        self.low_iou_threshold = 0.15   # For tentative tracks
        self.detection_threshold = 0.25  # High confidence detections (lowered for better low-quality detection)
        
    def cleanup_old_tracks(self):
        """Remove very old tracks to prevent memory buildup."""
        tracks_to_delete = []
        for track_id, track in self.tracks.items():
            if track.should_delete():
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            
        if tracks_to_delete:
            print(f"Cleaned up {len(tracks_to_delete)} old tracks")
    
    def get_dynamic_threshold(self, detection_conf: float, track_quality: float) -> float:
        """Get dynamic IoU threshold based on detection and track quality."""
        base_threshold = self.high_iou_threshold
        
        # Lower threshold for high-quality combinations
        if detection_conf > self.detection_threshold and track_quality > 0.7:
            return base_threshold - 0.1
        
        # Higher threshold for low-quality combinations
        if detection_conf < 0.4 or track_quality < 0.3:
            return base_threshold + 0.1
            
        return base_threshold
    
    def update(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """Update tracker with new detections using enhanced association."""
        # Predict all existing tracks
        for track in self.tracks.values():
            track.predict()
        
        if not detections:
            # No detections - mark all tracks as missed
            for track in self.tracks.values():
                track.mark_missed()
            
            # Return confirmed tracks only (including interpolated)
            return self._get_output_tracks()
        
        # Separate high and low confidence detections
        high_conf_detections = [d for d in detections if d.get('confidence', 0) >= self.detection_threshold]
        low_conf_detections = [d for d in detections if d.get('confidence', 0) < self.detection_threshold]
        
        # Associate high confidence detections first with confirmed tracks
        confirmed_tracks = {tid: track for tid, track in self.tracks.items() if track.state == 'confirmed'}
        tentative_tracks = {tid: track for tid, track in self.tracks.items() if track.state == 'tentative'}
        
        matched_track_ids = set()
        matched_detection_indices = set()
        
        # Stage 1: Match high confidence detections with confirmed tracks
        if high_conf_detections and confirmed_tracks:
            track_ids = list(confirmed_tracks.keys())
            track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
            det_bboxes = [d['bbox'] for d in high_conf_detections]
            
            matches = self._associate_detections_to_tracks(
                det_bboxes, track_bboxes, 
                [d.get('confidence', 0) for d in high_conf_detections],
                [self.tracks[tid].get_quality_score() for tid in track_ids]
            )
            
            for det_idx, track_idx in matches:
                track_id = track_ids[track_idx]
                detection = high_conf_detections[det_idx]
                self.tracks[track_id].update(detection['bbox'], detection, frame_number, self.fps)
                matched_track_ids.add(track_id)
                matched_detection_indices.add(det_idx)
        
        # Stage 2: Match remaining high confidence detections with tentative tracks
        remaining_high_conf = [d for i, d in enumerate(high_conf_detections) if i not in matched_detection_indices]
        remaining_tentative = {tid: track for tid, track in tentative_tracks.items() if tid not in matched_track_ids}
        
        if remaining_high_conf and remaining_tentative:
            track_ids = list(remaining_tentative.keys())
            track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
            det_bboxes = [d['bbox'] for d in remaining_high_conf]
            
            matches = self._associate_detections_to_tracks(
                det_bboxes, track_bboxes,
                [d.get('confidence', 0) for d in remaining_high_conf],
                [self.tracks[tid].get_quality_score() for tid in track_ids],
                threshold_offset=-0.05  # Slightly more lenient for tentative
            )
            
            for det_idx, track_idx in matches:
                # Find original index in high_conf_detections
                original_det_idx = next(i for i, d in enumerate(high_conf_detections) 
                                      if i not in matched_detection_indices and d == remaining_high_conf[det_idx])
                
                track_id = track_ids[track_idx]
                detection = remaining_high_conf[det_idx]
                self.tracks[track_id].update(detection['bbox'], detection, frame_number, self.fps)
                matched_track_ids.add(track_id)
                matched_detection_indices.add(original_det_idx)
        
        # Stage 3: Create new tracks for unmatched high confidence detections
        for i, detection in enumerate(high_conf_detections):
            if i not in matched_detection_indices:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = Track(track_id, detection['bbox'], detection)
                self.tracks[track_id].first_frame = frame_number
                self.tracks[track_id].last_frame = frame_number
        
        # Stage 4: Try to match low confidence detections with remaining tracks (more lenient)
        remaining_tracks = {tid: track for tid, track in self.tracks.items() if tid not in matched_track_ids}
        
        if low_conf_detections and remaining_tracks:
            track_ids = list(remaining_tracks.keys())
            track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
            det_bboxes = [d['bbox'] for d in low_conf_detections]
            
            matches = self._associate_detections_to_tracks(
                det_bboxes, track_bboxes,
                [d.get('confidence', 0) for d in low_conf_detections],
                [self.tracks[tid].get_quality_score() for tid in track_ids],
                threshold_offset=-0.1  # More lenient for low confidence
            )
            
            for det_idx, track_idx in matches:
                track_id = track_ids[track_idx]
                detection = low_conf_detections[det_idx]
                self.tracks[track_id].update(detection['bbox'], detection, frame_number, self.fps)
                matched_track_ids.add(track_id)
        
        # Mark unmatched tracks as missed
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.mark_missed()
        
        # Clean up tracks that should be deleted
        self.cleanup_old_tracks()
        
        return self._get_output_tracks()
    
    def _associate_detections_to_tracks(self, detection_bboxes: List[List[float]], 
                                      track_bboxes: List[List[float]],
                                      detection_confs: List[float],
                                      track_qualities: List[float],
                                      threshold_offset: float = 0.0) -> List[Tuple[int, int]]:
        """Enhanced association with dynamic thresholds."""
        if not detection_bboxes or not track_bboxes:
            return []
        
        import numpy as np
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detection_bboxes), len(track_bboxes)))
        
        for i, det_bbox in enumerate(detection_bboxes):
            for j, track_bbox in enumerate(track_bboxes):
                try:
                    iou = calculate_iou(det_bbox, track_bbox)
                    # Apply dynamic threshold
                    dynamic_threshold = self.get_dynamic_threshold(
                        detection_confs[i], track_qualities[j]
                    ) + threshold_offset
                    
                    # Only consider matches above dynamic threshold
                    if iou >= dynamic_threshold:
                        iou_matrix[i, j] = iou
                    else:
                        iou_matrix[i, j] = 0
                except (IndexError, ValueError) as e:
                    print(f"IoU calculation error: {e}")
                    iou_matrix[i, j] = 0
        
        # Find optimal assignment using Hungarian algorithm (greedy approximation)
        matches = []
        used_detections = set()
        used_tracks = set()
        
        # Sort by IoU value (highest first) for greedy assignment
        potential_matches = []
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] > 0:
                    potential_matches.append((iou_matrix[i, j], i, j))
        
        potential_matches.sort(reverse=True)  # Highest IoU first
        
        for iou_val, det_idx, track_idx in potential_matches:
            if det_idx not in used_detections and track_idx not in used_tracks:
                matches.append((det_idx, track_idx))
                used_detections.add(det_idx)
                used_tracks.add(track_idx)
        
        return matches
    
    def _get_output_tracks(self) -> List[Dict]:
        """Get tracks for output, including interpolated tracks."""
        output_tracks = []
        
        for track in self.tracks.values():
            # Include confirmed tracks and recent tentative tracks
            if track.state == 'confirmed' or (track.state == 'tentative' and track.time_since_update <= 2):
                # Determine if this is interpolated (predicted position)
                interpolated = track.time_since_update > 0
                output_tracks.append(track.to_dict(interpolated=interpolated))
        
        return output_tracks


class YOLOService:
    """Service for YOLOv8 inference and PPE compliance checking."""
    
    def __init__(self, model_path: str = "backend/models/best.pt"):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Dataset classes according to user specification
        self.all_class_names = ['Earplug', 'Gloves', 'Goggles', 'Helmet', 'Mask', 'Person', 'Shoes', 'Vest']
        self.class_names = []  # Will be populated from model
        self.target_classes = []  # Will be populated from model
        self.load_model()
        
        # Initialize tracker as None - will be created when needed for video
        self.tracker = None
        
        # Adaptive detection settings
        self.adaptive_threshold_enabled = True
        self.min_conf_threshold = 0.15  # Lower minimum for low-quality videos
        self.max_conf_threshold = 0.35  # Higher maximum for high-quality videos
        
        # Performance optimization settings
        self.max_video_duration = 600  # 10 minutes max processing time
        self.frame_skip_threshold = 1800  # Skip frames if video > 30 fps * 60 seconds
        self.adaptive_frame_skip = True
        
        # Standing detection settings
        self.enable_standing_detection = True
        self.standing_height_ratio_threshold = 1.2  # Height/width ratio for standing detection (lowered for better detection)
        self.standing_area_threshold = 0.02  # Minimum area ratio of image for standing detection (lowered for distant people)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load YOLOv8 model."""
        if model_path:
            self.model_path = model_path
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Get actual class names from the model
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
                print(f"Model class names: {self.class_names}")
            else:
                print(f"Using default class names: {self.class_names}")
            
            # Update target classes to include all PPE items from the model
            # All classes except Person are PPE items
            self.target_classes = self.class_names.copy()
            
            print(f"Target classes found in model: {self.target_classes}")
            print(f"Model loaded on {self.device}: {self.model_path}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def _analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze image quality metrics to determine detection parameters."""
        # Calculate Laplacian variance (sharpness/blur metric)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Estimate noise level
        noise = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        
        return {
            'sharpness': laplacian_var,
            'brightness': brightness,
            'contrast': contrast,
            'noise': noise
        }
    
    def _get_adaptive_confidence_threshold(self, image_quality: Dict[str, float], base_threshold: float = 0.25) -> float:
        """Calculate adaptive confidence threshold based on image quality."""
        if not self.adaptive_threshold_enabled:
            return base_threshold
        
        # Normalize quality metrics
        sharpness_score = min(1.0, image_quality['sharpness'] / 100.0)  # Good sharpness > 100
        brightness_score = 1.0 - abs(image_quality['brightness'] - 127.5) / 127.5  # Ideal brightness ~127.5
        contrast_score = min(1.0, image_quality['contrast'] / 50.0)  # Good contrast > 50
        noise_score = max(0.0, 1.0 - image_quality['noise'] / 10.0)  # Low noise < 10
        
        # Combine scores
        quality_score = (sharpness_score + brightness_score + contrast_score + noise_score) / 4.0
        
        # Adjust threshold based on quality (lower quality = lower threshold)
        if quality_score < 0.3:  # Very low quality
            adaptive_threshold = self.min_conf_threshold
        elif quality_score < 0.6:  # Low quality
            adaptive_threshold = base_threshold * 0.7
        elif quality_score > 0.8:  # High quality
            adaptive_threshold = min(self.max_conf_threshold, base_threshold * 1.2)
        else:  # Medium quality
            adaptive_threshold = base_threshold
        
        return adaptive_threshold
    
    def _preprocess_for_detection(self, image: np.ndarray, quality_metrics: Dict[str, float], fast_mode: bool = False) -> np.ndarray:
        """Apply preprocessing to improve detection on low-quality images."""
        processed = image.copy()
        
        # Skip expensive operations in fast mode
        if fast_mode:
            # Only apply essential brightness correction
            if quality_metrics.get('brightness', 127) < 80:
                processed = cv2.convertScaleAbs(processed, alpha=1.3, beta=20)
            elif quality_metrics.get('brightness', 127) > 180:
                processed = cv2.convertScaleAbs(processed, alpha=0.8, beta=-10)
            return processed
        
        # Apply CLAHE for low contrast images (faster version)
        if quality_metrics.get('contrast', 50) < 25:
            if len(processed.shape) == 3:
                # Faster: apply only to grayscale
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                # Blend with original
                processed = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
        
        # Skip expensive denoising in video processing
        
        # Apply light sharpening for very blurry images only
        if quality_metrics.get('sharpness', 100) < 30:
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        # Fast brightness adjustment
        brightness = quality_metrics.get('brightness', 127)
        if brightness < 80:
            processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=15)
        elif brightness > 180:
            processed = cv2.convertScaleAbs(processed, alpha=0.9, beta=-5)
        
        return processed
    
    def _run_multiscale_detection(self, image: np.ndarray, conf_threshold: float, iou_threshold: float) -> List:
        """Run detection at multiple scales to catch small/distant objects."""
        all_results = []
        
        # Original scale
        results_original = self.model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        all_results.extend(results_original)
        
        # Only do multiscale if image is large enough and we have low detections
        if image.shape[0] > 640 or image.shape[1] > 640:
            # Try at 1.2x scale (upscale for better small object detection)
            height, width = image.shape[:2]
            new_height, new_width = int(height * 1.2), int(width * 1.2)
            
            if new_height < 2000 and new_width < 2000:  # Avoid memory issues
                upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                results_upscaled = self.model(upscaled, conf=conf_threshold * 0.9, iou=iou_threshold, verbose=False)
                
                # Scale bounding boxes back to original size
                for result in results_upscaled:
                    if result.boxes is not None:
                        result.boxes.xyxy /= 1.2  # Scale back coordinates
                
                all_results.extend(results_upscaled)
        
        return all_results
    
    def _calculate_optimal_frame_skip(self, total_frames: int, fps: int, duration_seconds: float) -> int:
        """Calculate optimal frame skip based on video properties."""
        if not self.adaptive_frame_skip:
            return 1  # Process every frame
            
        # For very long videos, skip more frames
        if duration_seconds > 300:  # > 5 minutes
            return max(2, fps // 10)  # Process ~10 frames per second
        elif duration_seconds > 120:  # > 2 minutes  
            return max(1, fps // 15)  # Process ~15 frames per second
        elif total_frames > self.frame_skip_threshold:
            return max(1, fps // 20)  # Process ~20 frames per second
        else:
            return 1  # Process every frame for short videos
    
    def _detect_standing_posture(self, bbox: List[float], image_shape: Tuple[int, int]) -> Dict[str, Union[bool, float]]:
        """Detect if a person is standing based on bounding box geometry."""
        if not self.enable_standing_detection:
            return {'is_standing': False, 'confidence': 0.0, 'height_ratio': 0.0}
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Calculate height-to-width ratio
        height_ratio = height / width if width > 0 else 0
        
        # Calculate bounding box area relative to image
        bbox_area = width * height
        image_area = image_shape[0] * image_shape[1]  # height * width
        area_ratio = bbox_area / image_area if image_area > 0 else 0
        
        # Adjusted thresholds for better detection
        # Lower the height ratio threshold for better detection
        is_tall_enough = height_ratio >= 1.2  # Lowered from 1.5
        # Lower the area threshold for smaller/distant people
        is_large_enough = area_ratio >= 0.02  # Lowered from 0.1
        
        # Additional heuristics for better detection
        # Person should be touching or near the bottom of the frame (standing on ground)
        bottom_proximity = (image_shape[0] - y2) / image_shape[0]  # Distance from bottom
        is_grounded = bottom_proximity < 0.5  # Within 50% of bottom (more lenient)
        
        # Standing confidence based on multiple factors
        standing_confidence = 0.0
        if is_tall_enough:
            standing_confidence += 0.4
        if is_large_enough:
            standing_confidence += 0.3
        if is_grounded:
            standing_confidence += 0.3
        
        is_standing = standing_confidence >= 0.5  # Lowered threshold for better detection
        
        return {
            'is_standing': is_standing,
            'confidence': standing_confidence,
            'height_ratio': height_ratio,
            'area_ratio': area_ratio,
            'bottom_proximity': bottom_proximity
        }
    
    def _parse_detections(self, results, debug=False) -> Dict[str, List[Dict]]:
        """Parse YOLO results into categorized detections."""
        detections_by_class = {
            'Person': [],
            'Helmet': [],
            'Vest': [],
            'Earplug': [],
            'Gloves': [],
            'Goggles': [],
            'Mask': [],
            'Shoes': []
        }
        total_detections = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                total_detections += len(boxes)
                if debug:
                    print(f"YOLO detected {len(boxes)} total objects")
                
                for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(float)
                    confidence = float(conf.cpu().numpy())
                    class_id = int(cls.cpu().numpy())
                    
                    # Map class_id to class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
                    
                    if debug:
                        print(f"Detection {i}: class_id={class_id}, class_name='{class_name}', conf={confidence:.3f}")
                    
                    # Only process target classes
                    if class_name not in self.target_classes:
                        if debug:
                            print(f"Skipping {class_name} (not in target classes: {self.target_classes})")
                        continue
                    
                    # Create detection object
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id
                    }
                    
                    # Categorize by class name
                    if class_name in detections_by_class:
                        detections_by_class[class_name].append(detection)
            else:
                if debug:
                    print("No boxes detected by YOLO")
        
        if debug:
            print(f"Total YOLO detections: {total_detections}")
            for class_name, dets in detections_by_class.items():
                if dets:
                    print(f"Parsed: {len(dets)} {class_name}(s)")
        
        return detections_by_class
    
    def associate_ppe(self, detections_by_class: Dict[str, List[Dict]], 
                     required_ppe: List[str] = None, image_shape: Tuple[int, int] = None) -> List[Dict]:
        """Associate PPE items with persons and determine safety compliance."""
        person_statuses = []
        persons = detections_by_class.get('Person', [])
        
        for person in persons:
            person_bbox = person['bbox']
            ppe_detected = {}
            
            # Check each PPE type for containment within person bounding box
            for ppe_class in ['Helmet', 'Vest', 'Gloves', 'Goggles', 'Earplug', 'Mask', 'Shoes']:
                has_ppe = False
                ppe_items = detections_by_class.get(ppe_class, [])
                
                for ppe_item in ppe_items:
                    ppe_bbox = ppe_item['bbox']
                    containment_ratio = calculate_containment_ratio(ppe_bbox, person_bbox)
                    
                    # PPE is associated if significantly contained within person bbox
                    if containment_ratio > 0.3:  # 30% containment threshold
                        has_ppe = True
                        break
                
                ppe_detected[ppe_class] = has_ppe
            
            # Determine compliance status based on required PPE
            if required_ppe is None or len(required_ppe) == 0:
                # Default behavior - require helmet and vest
                status = "Safe" if (ppe_detected.get('Helmet', False) and ppe_detected.get('Vest', False)) else "Unsafe"
            else:
                # Check if all required PPE items are present
                all_required_present = all(ppe_detected.get(ppe, False) for ppe in required_ppe)
                status = "Safe" if all_required_present else "Unsafe"
            
            # Detect standing posture if image shape is provided
            standing_info = {'is_standing': False, 'confidence': 0.0}
            if image_shape is not None:
                standing_info = self._detect_standing_posture(person_bbox, image_shape)
            
            person_status = {
                'bbox': person_bbox,
                'confidence': person['confidence'],
                'status': status,
                'ppe_detected': ppe_detected,
                # Keep backward compatibility
                'has_helmet': ppe_detected.get('Helmet', False),
                'has_vest': ppe_detected.get('Vest', False),
                # Standing detection
                'is_standing': standing_info['is_standing'],
                'standing_confidence': standing_info['confidence']
            }
            
            person_statuses.append(person_status)
        
        return person_statuses
    
    def _draw_ppe_boxes(self, image: np.ndarray, detections_by_class: Dict[str, List[Dict]]):
        """Draw bounding boxes for individual PPE items."""
        # Define colors for different PPE items
        ppe_colors = {
            'Helmet': (0, 255, 255),      # Yellow
            'Vest': (255, 165, 0),        # Orange  
            'Gloves': (255, 0, 255),      # Magenta
            'Goggles': (0, 255, 0),       # Green
            'Earplug': (255, 255, 0),     # Cyan
            'Mask': (128, 0, 128),        # Purple
            'Shoes': (165, 42, 42)        # Brown
        }
        
        for ppe_class, detections in detections_by_class.items():
            if ppe_class == 'Person' or not detections:  # Skip person class
                continue
                
            color = ppe_colors.get(ppe_class, (255, 255, 255))  # Default white
            
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                confidence = detection['confidence']
                
                # Draw PPE bounding box with thinner lines
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                
                # Draw small label for PPE item - just abbreviation
                # Map class names to abbreviations
                ppe_abbrev = {'Helmet': 'H', 'Vest': 'V', 'Gloves': 'G', 'Goggles': 'Gg', 
                             'Earplug': 'E', 'Mask': 'M', 'Shoes': 'S'}
                label = ppe_abbrev.get(ppe_class, ppe_class[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                
                (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(image, 
                            (x1, y1 - label_height - 5),
                            (x1 + label_width + 5, y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(image, label, (x1 + 2, y1 - 2),
                          font, font_scale, (0, 0, 0), thickness)  # Black text
    
    def _draw_annotations(self, image: np.ndarray, person_statuses: List[Dict], detections_by_class: Dict[str, List[Dict]] = None) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        annotated_image = image.copy()
        
        # First, draw individual PPE item bounding boxes if detections are provided
        if detections_by_class:
            self._draw_ppe_boxes(annotated_image, detections_by_class)
        
        # Draw person count summary
        total_people = len(person_statuses)
        safe_count = sum(1 for p in person_statuses if p['status'] == 'Safe')
        unsafe_count = sum(1 for p in person_statuses if p['status'] == 'Unsafe')
        
        count_text = f"People: {total_people} (Safe: {safe_count}, Unsafe: {unsafe_count})"
        cv2.putText(annotated_image, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Then draw person bounding boxes and labels
        for i, person in enumerate(person_statuses):
            bbox = person['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            confidence = person['confidence']
            status = person['status']
            
            # Color based on PPE safety status only
            is_standing = person.get('is_standing', False)
            
            if status == "Safe" and not is_standing:
                color = (0, 255, 0)  # Green for safe and not standing
                bg_color = (0, 200, 0)
            else:
                # Red for unsafe (either missing PPE or standing)
                color = (0, 0, 255)  # Red for unsafe
                bg_color = (0, 0, 200)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Build PPE status string dynamically - show only detected items
            ppe_detected = person.get('ppe_detected', {})
            detected_items = []
            for ppe, symbol in [('Helmet', 'H'), ('Vest', 'V'), ('Gloves', 'G'), ('Goggles', 'Gg'), ('Earplug', 'E'), ('Mask', 'M'), ('Shoes', 'S')]:
                if ppe_detected.get(ppe, False):
                    detected_items.append(symbol)
            
            # Create single simplified label with Person, PPE items, and standing status
            standing_indicator = ""
            if person.get('is_standing', False):
                standing_indicator = " [Standing]"
            
            if detected_items:
                label = f"Person: {','.join(detected_items)}{standing_indicator}"
            else:
                # Show status if no specific PPE detected
                status = person.get('status', 'Unknown')
                label = f"Person ({status}){standing_indicator}"
            
            # Calculate text size for simplified label - make more readable
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw smaller label background
            label_bg_height = label_height + 6
            label_bg_width = label_width + 8
            
            # Ensure label stays within image bounds
            label_x = max(x1, 0)
            label_y = max(y1 - label_bg_height, 0)
            
            cv2.rectangle(annotated_image, 
                         (label_x, label_y),
                         (label_x + label_bg_width, label_y + label_bg_height),
                         bg_color, -1)
            
            # Draw simplified text - Person with PPE items
            text_y = label_y + label_height + 3
            
            cv2.putText(annotated_image, label, (label_x + 4, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        return annotated_image
    
    def _draw_video_annotations(self, image: np.ndarray, tracked_people: List[Dict], 
                               frame_number: int, total_frames: int, detections_by_class: Dict[str, List[Dict]] = None) -> np.ndarray:
        """Draw enhanced annotations for video with tracking info."""
        annotated_image = image.copy()
        
        # First, draw individual PPE item bounding boxes if detections are provided
        if detections_by_class:
            self._draw_ppe_boxes(annotated_image, detections_by_class)
        
        # Draw frame info with accurate person count
        total_people = len(tracked_people)
        real_people = len([p for p in tracked_people if not p.get('interpolated', False)])
        safe_count = sum(1 for p in tracked_people if p['status'] == 'Safe' and not p.get('interpolated', False))
        unsafe_count = sum(1 for p in tracked_people if p['status'] == 'Unsafe' and not p.get('interpolated', False))
        
        frame_text = f"Frame: {frame_number}/{total_frames}"
        count_text = f"People: {real_people} (Safe: {safe_count}, Unsafe: {unsafe_count})"
        
        # Draw frame number
        cv2.putText(annotated_image, frame_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw people count
        cv2.putText(annotated_image, count_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, person in enumerate(tracked_people):
            bbox = person['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            confidence = person['confidence']
            status = person['status']
            is_interpolated = person.get('interpolated', False)
            
            # Color based on PPE safety status only, with interpolation shading
            is_standing = person.get('is_standing', False)
            
            if status == "Safe" and not is_standing:
                # Green only for safe and not standing
                if is_interpolated:
                    color = (0, 180, 0)  # Darker green for interpolated
                    bg_color = (0, 150, 0)
                else:
                    color = (0, 255, 0)  # Bright green for detected
                    bg_color = (0, 200, 0)
            else:
                # Red for unsafe (either missing PPE or standing)
                if is_interpolated:
                    color = (0, 0, 180)  # Darker red for interpolated
                    bg_color = (0, 0, 150)
                else:
                    color = (0, 0, 255)  # Bright red for detected
                    bg_color = (0, 0, 200)
            
            thickness = 2 if is_interpolated else 3  # Thinner lines for interpolated
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Build PPE status string dynamically - show only detected items
            ppe_detected = person.get('ppe_detected', {})
            detected_items = []
            for ppe, symbol in [('Helmet', 'H'), ('Vest', 'V'), ('Gloves', 'G'), ('Goggles', 'Gg'), ('Earplug', 'E'), ('Mask', 'M'), ('Shoes', 'S')]:
                if ppe_detected.get(ppe, False):
                    detected_items.append(symbol)
            
            # Create single simplified label with Person, PPE items, and standing status
            standing_indicator = ""
            if person.get('is_standing', False):
                standing_time = person.get('total_standing_time', 0)
                standing_indicator = f" [Standing:{standing_time:.1f}s]"
            
            if detected_items:
                label = f"Person: {','.join(detected_items)}{standing_indicator}"
            else:
                # Show status if no specific PPE detected
                status = person.get('status', 'Unknown')
                label = f"Person ({status}){standing_indicator}"
            
            # Calculate text size for simplified label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness_text = 2
            
            (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness_text)
            
            # Draw smaller label background
            label_bg_height = label_height + 6
            label_bg_width = label_width + 8
            
            # Ensure label stays within image bounds
            label_x = max(x1, 0)
            label_y = max(y1 - label_bg_height, 0)
            
            cv2.rectangle(annotated_image, 
                         (label_x, label_y),
                         (label_x + label_bg_width, label_y + label_bg_height),
                         bg_color, -1)
            
            # Draw simplified text - Person with PPE items
            text_y = label_y + label_height + 3
            
            cv2.putText(annotated_image, label, (label_x + 4, text_y),
                       font, font_scale, (255, 255, 255), thickness_text)
        
        return annotated_image
    
    def infer_image(self, 
                   image_bytes: bytes, 
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   save_outputs: bool = False,
                   output_dir: str = "outputs",
                   required_ppe: List[str] = None) -> Tuple[bytes, Dict]:
        """
        Run inference on image and return annotated image bytes and results.
        """
        # Convert bytes to PIL Image
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image_pil)
        
        # Analyze image quality and adjust parameters
        quality_metrics = self._analyze_image_quality(image_np)
        adaptive_conf = self._get_adaptive_confidence_threshold(quality_metrics, conf_threshold)
        
        # Preprocess image if low quality
        preprocessed_image = self._preprocess_for_detection(image_np, quality_metrics)
        
        # Run YOLO inference with adaptive parameters
        results = self.model(preprocessed_image, conf=adaptive_conf, iou=iou_threshold, verbose=False)
        
        # If very few detections and low quality, try multiscale detection
        initial_detections = sum(len(result.boxes) if result.boxes is not None else 0 for result in results)
        if initial_detections < 2 and quality_metrics['sharpness'] < 100:
            print(f"Low detections ({initial_detections}) detected, trying multiscale detection...")
            results = self._run_multiscale_detection(preprocessed_image, adaptive_conf, iou_threshold)
        
        # Parse detections
        detections_by_class = self._parse_detections(results)
        
        # Associate PPE with persons (include image shape for standing detection)
        person_statuses = self.associate_ppe(detections_by_class, required_ppe, preprocessed_image.shape[:2])
        
        # Draw annotations
        annotated_image = self._draw_annotations(image_np, person_statuses, detections_by_class)
        
        # Convert back to bytes
        annotated_pil = Image.fromarray(annotated_image)
        img_bytes = io.BytesIO()
        annotated_pil.save(img_bytes, format='JPEG')
        annotated_bytes = img_bytes.getvalue()
        
        # Save output if requested
        output_path = None
        if save_outputs:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "annotated_image.jpg")
            annotated_pil.save(output_path)
        
        # Prepare response JSON
        standing_count = sum(1 for p in person_statuses if p.get('is_standing', False))
        
        response_json = {
            "source": "image",
            "people": person_statuses,
            "counts": {
                "safe": sum(1 for p in person_statuses if p['status'] == 'Safe'),
                "unsafe": sum(1 for p in person_statuses if p['status'] == 'Unsafe'),
                "total": len(person_statuses),
                "standing": standing_count
            }
        }
        
        if output_path:
            response_json["output_path"] = output_path
        
        return annotated_bytes, response_json
    
    def infer_video(self,
                   video_path: str,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   save_outputs: bool = True,
                   output_dir: str = "outputs",
                   required_ppe: List[str] = None,
                   manual_frame_skip: Optional[int] = None) -> Tuple[Optional[str], Dict]:
        """
        Run inference on video and return output video path and results.
        Enhanced with better frame processing and statistics.
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # Ensure fps is at least 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        print(f"DEBUG: Video FPS detected: {fps}")  # Debug FPS
        
        # Calculate optimal frame skip for performance
        if manual_frame_skip is not None:
            frame_skip = manual_frame_skip
            print(f"Using manual frame skip: {frame_skip}")
        else:
            frame_skip = self._calculate_optimal_frame_skip(total_frames, fps, duration_seconds)
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames, {duration_seconds:.1f}s")
        print(f"Frame skip: {frame_skip} (processing every {frame_skip} frame{'s' if frame_skip > 1 else ''})")
        
        # Check if video is too long and warn
        if duration_seconds > self.max_video_duration:
            print(f"WARNING: Video duration ({duration_seconds:.1f}s) exceeds recommended maximum ({self.max_video_duration}s)")
            print(f"Consider using frame_skip={max(frame_skip, fps//5)} for faster processing")
        
        # Setup output video if saving
        output_path = None
        out = None
        if save_outputs:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "annotated_video.mp4")
            # Use H.264 codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("Warning: Could not initialize video writer")
        
        # Use existing tracker or create new one if not exists (maintain continuity)
        if not hasattr(self, 'tracker') or self.tracker is None:
            self.tracker = MultiObjectTracker(max_disappeared=10, max_distance=150, fps=fps)
        else:
            # Update existing tracker's fps to match current video
            self.tracker.fps = fps
        # Optional: Clear very old tracks to prevent memory issues
        self.tracker.cleanup_old_tracks()
        frame_stats = []
        frame_count = 0
        processed_frames = 0
        last_tracked_people = []  # Store last processed tracking results for skipped frames
        last_detections_by_class = {}  # Store last detections for skipped frames
        
        print(f"Processing frames with skip={frame_skip} for optimal performance ({fps} FPS)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames based on calculated skip rate
            if frame_skip > 1 and (frame_count - 1) % frame_skip != 0:
                # Apply last known annotations to skipped frame to prevent blinking
                if out is not None:
                    if last_tracked_people:  # Use last known tracking results
                        skipped_frame = self._draw_video_annotations(
                            frame, last_tracked_people, frame_count, total_frames, last_detections_by_class
                        )
                        out.write(skipped_frame)
                    else:
                        out.write(frame)  # First few frames before any tracking
                continue
            
            try:
                # Analyze frame quality less frequently for long videos
                analysis_interval = 30 if duration_seconds > 120 else 10
                if processed_frames % analysis_interval == 0:  # Analyze quality periodically
                    self.current_quality_metrics = self._analyze_image_quality(frame)
                    self.current_adaptive_conf = self._get_adaptive_confidence_threshold(
                        self.current_quality_metrics, conf_threshold
                    )
                    
                    # Log quality info for debugging
                    if processed_frames <= 3 or processed_frames % 100 == 0:
                        quality_score = (
                            min(1.0, self.current_quality_metrics['sharpness'] / 100.0) +
                            min(1.0, self.current_quality_metrics['contrast'] / 50.0)
                        ) / 2.0
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        print(f"Frame {frame_count} ({progress:.1f}%): Quality score: {quality_score:.2f}, "
                              f"Adaptive conf: {self.current_adaptive_conf:.3f}")
                
                # Use cached values if available
                adaptive_conf = getattr(self, 'current_adaptive_conf', conf_threshold)
                quality_metrics = getattr(self, 'current_quality_metrics', {})
                
                # Fast preprocessing for videos (only essential corrections)
                if quality_metrics and (quality_metrics.get('sharpness', 100) < 40 or 
                                       quality_metrics.get('brightness', 127) < 70 or
                                       quality_metrics.get('brightness', 127) > 200):
                    processed_frame = self._preprocess_for_detection(frame, quality_metrics, fast_mode=True)
                else:
                    processed_frame = frame
                
                # Run inference on frame
                results = self.model(processed_frame, conf=adaptive_conf, iou=iou_threshold, verbose=False)
                
                # Parse detections with debug info on first few frames
                debug_this_frame = frame_count <= 3 or frame_count % 30 == 0
                detections_by_class = self._parse_detections(results, debug=debug_this_frame)
                
                # Debug: Log detection counts every 30 frames
                if frame_count % 30 == 0:
                    persons_count = len(detections_by_class.get('Person', []))
                    print(f"Frame {frame_count}: Found {persons_count} persons")
                
                # Associate PPE with persons (include frame shape for standing detection)
                person_statuses = self.associate_ppe(detections_by_class, required_ppe, processed_frame.shape[:2])
                
                # Debug: Log standing detection
                standing_people = [p for p in person_statuses if p.get('is_standing', False)]
                if standing_people and frame_count % 30 == 0:
                    print(f"DEBUG: Frame {frame_count}: Found {len(standing_people)} standing people")
                
                # Debug: Log person statuses
                if frame_count % 30 == 0 and person_statuses:
                    print(f"Frame {frame_count}: Person statuses: {len(person_statuses)} people detected")
                
                # Update tracker with new detections (handle empty detections)
                if person_statuses:
                    tracked_people = self.tracker.update(person_statuses, frame_count)
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: Tracker returned {len(tracked_people)} tracked people")
                else:
                    # No detections this frame - just predict existing tracks
                    tracked_people = self.tracker.update([], frame_count)
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: No detections, tracker returned {len(tracked_people)} tracked people")
                
                # Draw annotations with frame info using tracked people
                annotated_frame = self._draw_video_annotations(frame, tracked_people, frame_count, total_frames, detections_by_class)
                
                # Store results for skipped frame annotations (prevent blinking)
                last_tracked_people = tracked_people.copy()
                last_detections_by_class = detections_by_class.copy() if detections_by_class else {}
                
                # Track frame statistics 
                current_people_count = len([p for p in tracked_people if not p.get('interpolated', False)])
                frame_stat = {
                    'frame_number': frame_count,
                    'people_count': current_people_count,
                    'safe_count': sum(1 for p in tracked_people if p['status'] == 'Safe' and not p.get('interpolated', False)),
                    'unsafe_count': sum(1 for p in tracked_people if p['status'] == 'Unsafe' and not p.get('interpolated', False)),
                }
                frame_stats.append(frame_stat)
                
                # Write frame if saving
                if out is not None:
                    out.write(annotated_frame)
                
                processed_frames += 1
                
                # Progress feedback for long videos - more frequent for long videos
                progress_interval = max(fps * 5, 150) if duration_seconds > 60 else fps * 10  # Every 5 seconds for long videos
                if frame_count % progress_interval == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    processing_speed = processed_frames / max(1, frame_count - 1) if frame_count > 1 else 1.0
                    est_remaining = (total_frames - frame_count) / (processing_speed * fps) if processing_speed > 0 and fps > 0 else 0
                    print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - "
                          f"Processing speed: {processing_speed:.2f}x - "
                          f"Est. remaining: {est_remaining:.1f}s")
                    
            except Exception as frame_error:
                print(f"Error processing frame {frame_count}: {frame_error}")
                # Write original frame on error
                if out is not None:
                    out.write(frame)
                continue
        
        # Clean up
        cap.release()
        if out is not None:
            out.release()
        
        print(f"Video processing completed: {processed_frames} frames processed out of {frame_count} total")
        
        # Calculate minimum appearances threshold - much more lenient
        min_appearances = max(1, processed_frames // 50)  # At least 2% of processed frames, minimum 1
        
        print(f"Total tracks in tracker: {len(self.tracker.tracks)}")
        print(f"Minimum appearances threshold: {min_appearances}")
        
        # Debug: Show all tracks and their appearance counts
        for track_id, track in self.tracker.tracks.items():
            print(f"Track {track_id}: {track.total_appearances} appearances, state: {track.state}")
        
        # Get final tracked people from the tracker and finalize standing times
        final_tracked_people = []
        for track in self.tracker.tracks.values():
            if track.total_appearances >= min_appearances:
                # Finalize standing time for each track
                final_time = track.finalize_standing_time(frame_count, fps)
                print(f"DEBUG: Track {track.track_id} finalized with {final_time:.2f}s standing time (FPS: {fps})")
                final_tracked_people.append(track)
        
        # Fallback: If no tracks meet threshold, include all confirmed tracks
        if len(final_tracked_people) == 0 and len(self.tracker.tracks) > 0:
            print("No tracks met minimum appearances threshold, using all confirmed tracks")
            final_tracked_people = []
            for track in self.tracker.tracks.values():
                if track.state == 'confirmed' or track.total_appearances >= 1:
                    # Finalize standing time for each track
                    final_time = track.finalize_standing_time(frame_count, fps)
                    print(f"DEBUG: Fallback track {track.track_id} finalized with {final_time:.2f}s standing time (FPS: {fps})")
                    final_tracked_people.append(track)
        
        print(f"ByteTracker found {len(final_tracked_people)} stable tracks across all frames")
        
        # Convert tracks to final people format
        final_people = []
        for track in final_tracked_people:
            person = {
                'person_id': track.track_id,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'status': track.status,
                'has_helmet': track.has_helmet,
                'has_vest': track.has_vest,
                'ppe_detected': track.ppe_detected,  # Include full PPE detection info
                'tracking_stats': {
                    'total_appearances': track.total_appearances,
                    'safe_appearances': track.safe_count,
                    'unsafe_appearances': track.unsafe_count,
                    'safety_percentage': track.get_safety_percentage(),
                    'first_seen_frame': track.first_frame,
                    'last_seen_frame': track.last_frame
                },
                'standing_stats': {
                    'is_standing': track.is_standing,
                    'total_standing_time': round(track.total_standing_time, 2),
                    'standing_percentage': round(track.get_standing_percentage(), 1),
                    'standing_confidence': track.standing_confidence
                }
            }
            final_people.append(person)
        
        # Calculate statistics based on tracked people
        total_unique_people = len(final_people)
        total_safe = sum(1 for p in final_people if p['status'] == 'Safe')
        total_unsafe = total_unique_people - total_safe
        
        # Calculate averages and trends from frame stats
        avg_people_per_frame = sum(stat['people_count'] for stat in frame_stats) / max(1, len(frame_stats))
        max_people_in_frame = max((stat['people_count'] for stat in frame_stats), default=0)
        safety_rate = (total_safe / max(1, total_unique_people)) * 100
        
        # Calculate standing statistics
        total_standing_people = sum(1 for p in final_people if p['standing_stats']['total_standing_time'] > 0)
        avg_standing_time = sum(p['standing_stats']['total_standing_time'] for p in final_people) / max(1, total_unique_people)
        max_standing_time = max((p['standing_stats']['total_standing_time'] for p in final_people), default=0)
        avg_standing_percentage = sum(p['standing_stats']['standing_percentage'] for p in final_people) / max(1, total_unique_people)
        
        response_json = {
            "source": "video",
            "people": final_people,  # Unique tracked people with tracking stats
            "counts": {
                "safe": total_safe,
                "unsafe": total_unsafe,
                "total": total_unique_people
            },
            "video_info": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "duration_seconds": total_frames / fps if fps > 0 else 0
            },
            "analytics": {
                "unique_people_detected": total_unique_people,
                "total_person_detections": len(self.tracker.tracks),  # Total tracks created
                "average_people_per_frame": round(avg_people_per_frame, 2),
                "max_people_in_frame": max_people_in_frame,
                "safety_compliance_rate": round(safety_rate, 1),
                "frames_with_detections": len([s for s in frame_stats if s['people_count'] > 0]),
                "processing_efficiency": round((processed_frames / frame_count) * 100, 1),
                "tracking_threshold": min_appearances,
                "standing_analytics": {
                    "people_who_stood": total_standing_people,
                    "average_standing_time": round(avg_standing_time, 2),
                    "max_standing_time": round(max_standing_time, 2),
                    "average_standing_percentage": round(avg_standing_percentage, 1),
                    "standing_detection_enabled": self.enable_standing_detection
                }
            }
        }
        
        if output_path:
            response_json["output_path"] = output_path
        
        return output_path, response_json
    
    def infer_frame(self, frame: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45, required_ppe: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Run inference on a single frame (for webcam streaming).
        """
        # Analyze frame quality and adapt parameters
        quality_metrics = self._analyze_image_quality(frame)
        adaptive_conf = self._get_adaptive_confidence_threshold(quality_metrics, conf_threshold)
        
        # Preprocess frame if needed
        processed_frame = self._preprocess_for_detection(frame, quality_metrics)
        
        # Run YOLO inference with adaptive parameters
        results = self.model(processed_frame, conf=adaptive_conf, iou=iou_threshold, verbose=False)
        
        # Parse detections
        detections_by_class = self._parse_detections(results)
        
        # Associate PPE with persons (include frame shape for standing detection)
        person_statuses = self.associate_ppe(detections_by_class, required_ppe, processed_frame.shape[:2])
        
        # Draw annotations
        annotated_frame = self._draw_annotations(frame, person_statuses, detections_by_class)
        
        # Prepare response JSON
        standing_count = sum(1 for p in person_statuses if p.get('is_standing', False))
        
        response_json = {
            "source": "webcam",
            "people": person_statuses,
            "counts": {
                "safe": sum(1 for p in person_statuses if p['status'] == 'Safe'),
                "unsafe": sum(1 for p in person_statuses if p['status'] == 'Unsafe'),
                "total": len(person_statuses),
                "standing": standing_count
            }
        }
        
        return annotated_frame, response_json


# Global model instance for FastAPI
yolo_service = YOLOService()