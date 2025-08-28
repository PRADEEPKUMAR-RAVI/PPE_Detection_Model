"""
Geometry utilities for bounding box operations.
"""
from typing import List, Tuple


def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x_center, y_center, width, height = bbox
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]


def xyxy_to_xywh(bbox: List[float]) -> List[float]:
    """Convert (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height)."""
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [x_center, y_center, width, height]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate intersection area
    if x2_i <= x1_i or y2_i <= y1_i:
        intersection = 0
    else:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union


def is_contained(inner_box: List[float], outer_box: List[float], margin: float = 0.0) -> bool:
    """
    Check if inner_box is contained within outer_box with optional margin.
    
    Args:
        inner_box: Inner bounding box [x1, y1, x2, y2]
        outer_box: Outer bounding box [x1, y1, x2, y2]
        margin: Margin for containment check (0.0 = strict containment)
        
    Returns:
        True if inner_box is contained within outer_box
    """
    x1_inner, y1_inner, x2_inner, y2_inner = inner_box
    x1_outer, y1_outer, x2_outer, y2_outer = outer_box
    
    # Apply margin to outer box (expand it)
    x1_outer -= margin
    y1_outer -= margin
    x2_outer += margin
    y2_outer += margin
    
    # Check if all corners of inner box are within outer box
    return (x1_inner >= x1_outer and 
            y1_inner >= y1_outer and 
            x2_inner <= x2_outer and 
            y2_inner <= y2_outer)


def calculate_containment_ratio(inner_box: List[float], outer_box: List[float]) -> float:
    """
    Calculate what percentage of inner_box is contained within outer_box.
    
    Args:
        inner_box: Inner bounding box [x1, y1, x2, y2]
        outer_box: Outer bounding box [x1, y1, x2, y2]
        
    Returns:
        Containment ratio between 0 and 1
    """
    x1_inner, y1_inner, x2_inner, y2_inner = inner_box
    x1_outer, y1_outer, x2_outer, y2_outer = outer_box
    
    # Calculate intersection coordinates
    x1_i = max(x1_inner, x1_outer)
    y1_i = max(y1_inner, y1_outer)
    x2_i = min(x2_inner, x2_outer)
    y2_i = min(y2_inner, y2_outer)
    
    # Calculate intersection area
    if x2_i <= x1_i or y2_i <= y1_i:
        intersection = 0
    else:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate inner box area
    inner_area = (x2_inner - x1_inner) * (y2_inner - y1_inner)
    
    # Avoid division by zero
    if inner_area == 0:
        return 0
    
    return intersection / inner_area


def get_box_center(bbox: List[float]) -> Tuple[float, float]:
    """Get the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_box_area(bbox: List[float]) -> float:
    """Calculate the area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)