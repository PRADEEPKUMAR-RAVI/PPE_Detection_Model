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
    
    def update(self, new_bbox: List[float], detection: Dict, frame_number: int):
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
    
    def to_dict(self, interpolated: bool = False) -> Dict:
        """Convert track to dictionary format."""
        return {
            'person_id': self.track_id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'status': self.status,
            'has_helmet': self.has_helmet,
            'has_vest': self.has_vest,
            'interpolated': interpolated,
            'quality_score': self.get_quality_score(),
            'total_appearances': self.total_appearances
        }


class MultiObjectTracker:
    """Enhanced multi-object tracker with ByteTrack-style lifecycle management."""
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 150):
        self.tracks = {}  # Dict of track_id -> Track
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Dynamic thresholds
        self.high_iou_threshold = 0.3  # For confirmed tracks
        self.low_iou_threshold = 0.2   # For tentative tracks
        self.detection_threshold = 0.4  # High confidence detections (lowered from 0.6)
        
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
                self.tracks[track_id].update(detection['bbox'], detection, frame_number)
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
                self.tracks[track_id].update(detection['bbox'], detection, frame_number)
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
                self.tracks[track_id].update(detection['bbox'], detection, frame_number)
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
                     required_ppe: List[str] = None) -> List[Dict]:
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
            
            person_status = {
                'bbox': person_bbox,
                'confidence': person['confidence'],
                'status': status,
                'ppe_detected': ppe_detected,
                # Keep backward compatibility
                'has_helmet': ppe_detected.get('Helmet', False),
                'has_vest': ppe_detected.get('Vest', False)
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
            
            # Color based on safety status
            if status == "Safe":
                color = (0, 255, 0)  # Green for safe
                bg_color = (0, 200, 0)
            else:
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
            
            # Create single simplified label with Person and PPE items
            if detected_items:
                label = f"Person: {','.join(detected_items)}"
            else:
                # Show status if no specific PPE detected
                status = person.get('status', 'Unknown')
                label = f"Person ({status})"
            
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
            
            # Color based on safety status with different shades for interpolated
            if status == "Safe":
                if is_interpolated:
                    color = (0, 180, 0)  # Darker green for interpolated
                    bg_color = (0, 150, 0)
                else:
                    color = (0, 255, 0)  # Bright green for detected
                    bg_color = (0, 200, 0)
            else:
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
            
            # Create single simplified label with Person and PPE items
            if detected_items:
                label = f"Person: {','.join(detected_items)}"
            else:
                # Show status if no specific PPE detected
                status = person.get('status', 'Unknown')
                label = f"Person ({status})"
            
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
        
        # Run YOLO inference
        results = self.model(image_np, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Parse detections
        detections_by_class = self._parse_detections(results)
        
        # Associate PPE with persons
        person_statuses = self.associate_ppe(detections_by_class, required_ppe)
        
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
        response_json = {
            "source": "image",
            "people": person_statuses,
            "counts": {
                "safe": sum(1 for p in person_statuses if p['status'] == 'Safe'),
                "unsafe": sum(1 for p in person_statuses if p['status'] == 'Unsafe'),
                "total": len(person_statuses)
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
                   required_ppe: List[str] = None) -> Tuple[Optional[str], Dict]:
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
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
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
            self.tracker = MultiObjectTracker(max_disappeared=10, max_distance=150)
        # Optional: Clear very old tracks to prevent memory issues
        self.tracker.cleanup_old_tracks()
        frame_stats = []
        frame_count = 0
        processed_frames = 0
        # NO FRAME SKIPPING - process every frame for stable tracking
        
        print(f"Processing ALL frames for maximum tracking stability ({fps} FPS)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process EVERY frame - no skipping
            
            try:
                # Run inference on frame
                results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                
                # Parse detections with debug info on first few frames
                debug_this_frame = frame_count <= 3 or frame_count % 30 == 0
                detections_by_class = self._parse_detections(results, debug=debug_this_frame)
                
                # Debug: Log detection counts every 30 frames
                if frame_count % 30 == 0:
                    persons_count = len(detections_by_class.get('Person', []))
                    print(f"Frame {frame_count}: Found {persons_count} persons")
                
                # Associate PPE with persons
                person_statuses = self.associate_ppe(detections_by_class, required_ppe)
                
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
                
                # Progress feedback for long videos
                if frame_count % (fps * 10) == 0:  # Every 10 seconds
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
                    
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
        
        # Get final tracked people from the tracker
        final_tracked_people = [
            track for track in self.tracker.tracks.values() 
            if track.total_appearances >= min_appearances
        ]
        
        # Fallback: If no tracks meet threshold, include all confirmed tracks
        if len(final_tracked_people) == 0 and len(self.tracker.tracks) > 0:
            print("No tracks met minimum appearances threshold, using all confirmed tracks")
            final_tracked_people = [
                track for track in self.tracker.tracks.values() 
                if track.state == 'confirmed' or track.total_appearances >= 1
            ]
        
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
                "tracking_threshold": min_appearances
            }
        }
        
        if output_path:
            response_json["output_path"] = output_path
        
        return output_path, response_json
    
    def infer_frame(self, frame: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45, required_ppe: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Run inference on a single frame (for webcam streaming).
        """
        # Run YOLO inference
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Parse detections
        detections_by_class = self._parse_detections(results)
        
        # Associate PPE with persons
        person_statuses = self.associate_ppe(detections_by_class, required_ppe)
        
        # Draw annotations
        annotated_frame = self._draw_annotations(frame, person_statuses, detections_by_class)
        
        # Prepare response JSON
        response_json = {
            "source": "webcam",
            "people": person_statuses,
            "counts": {
                "safe": sum(1 for p in person_statuses if p['status'] == 'Safe'),
                "unsafe": sum(1 for p in person_statuses if p['status'] == 'Unsafe'),
                "total": len(person_statuses)
            }
        }
        
        return annotated_frame, response_json


# Global model instance for FastAPI
yolo_service = YOLOService()