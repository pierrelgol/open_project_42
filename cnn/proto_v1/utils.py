import torch
import numpy as np
from typing import List
from PIL import Image, ImageDraw
import math


def get_nearest_power_of_2(size: int) -> int:
    """
    Calculate the nearest power of 2 that is >= size.

    Args:
        size: Input size

    Returns:
        Nearest power of 2 >= size
    """
    return 2 ** math.ceil(math.log2(size))


def resize_image_square(
    image: Image.Image, target_size: int = 416
) -> Image.Image:
    """
    Resize image to a fixed square size, stretching aspect ratio.

    Args:
        image: PIL Image to process
        target_size: Target square size (default 416)

    Returns:
        Resized square image
    """

    if image.mode != "L":
        image = image.convert("L")

    resized_image = image.resize(
        (target_size, target_size), Image.Resampling.LANCZOS
    )

    return resized_image


def yolo_to_pixel_coords(box: list, img_size: tuple) -> tuple:
    """
    Convert YOLO format [x_center, y_center, w, h] (normalized) to pixel
    coordinates (x1, y1, x2, y2).
    Args:
        box: [x_center, y_center, w, h] (all in [0, 1])
        img_size: (width, height)
    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    x_c, y_c, w, h = box
    img_w, img_h = img_size
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes(
    image: Image.Image, boxes: List[List[float]], color: str = "red"
) -> Image.Image:
    """
    Draw bounding boxes on a PIL image.
    Args:
        image: PIL Image (grayscale or RGB)
        boxes: List of [x_center, y_center, w, h] (YOLO format, normalized)
        color: Box color
    Returns:
        Image with boxes drawn
    """

    if image.mode == "L":
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixel_coords(box, (img_w, img_h))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return image


def iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def nms(
    boxes: List[List[float]], scores: List[float], iou_threshold: float = 0.5
) -> List[int]:
    """
    Apply Non-Maximum Suppression (NMS).
    Args:
        boxes: List of [x1, y1, x2, y2] (pixel coordinates)
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression
    Returns:
        List of indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    indices = scores_np.argsort()[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        rest = indices[1:]
        ious = np.array([iou(boxes_np[current], boxes_np[i]) for i in rest])
        indices = rest[ious <= iou_threshold]
    return keep


def save_checkpoint(model, filepath):
    """
    Save model checkpoint to file.

    Args:
        model: PyTorch model to save
        filepath: Path where to save the checkpoint
    """
    torch.save(model.state_dict(), filepath)


def load_checkpoint(model, filepath, device):
    """
    Load model checkpoint from file.

    Args:
        model: PyTorch model to load weights into
        filepath: Path to the checkpoint file
        device: Device to load the model on (cpu/cuda)
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
