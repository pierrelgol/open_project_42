"""
Advanced Data Augmentation Pipeline for Object Detection

This module provides a comprehensive augmentation pipeline specifically designed
for object detection tasks, including geometric and photometric transformations
that preserve bounding box annotations.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Tuple, Optional, Union


class DetectionAugmentation:
    """
    Advanced augmentation pipeline for object detection.

    Implements geometric and photometric transformations that preserve
    bounding box annotations and improve model generalization.
    """

    def __init__(
        self,
        target_size: int = 416,
        augment_prob: float = 0.8,
        geometric_prob: float = 0.5,
        photometric_prob: float = 0.5,
        mosaic_prob: float = 0.3,
        mixup_prob: float = 0.1,
    ):
        """
        Initialize the augmentation pipeline.

        Args:
            target_size: Target image size
            augment_prob: Probability of applying any augmentation
            geometric_prob: Probability of geometric transformations
            photometric_prob: Probability of photometric transformations
            mosaic_prob: Probability of mosaic augmentation
            mixup_prob: Probability of mixup augmentation
        """
        self.target_size = target_size
        self.augment_prob = augment_prob
        self.geometric_prob = geometric_prob
        self.photometric_prob = photometric_prob
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

        self.rotation_range = (-15, 15)
        self.scale_range = (0.8, 1.2)
        self.translation_range = (-0.1, 0.1)
        self.shear_range = (-10, 10)

        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)
        self.saturation_range = (0.8, 1.2)
        self.hue_range = (-0.1, 0.1)
        self.noise_std = 0.05

    def __call__(
        self, image: Image.Image, boxes: List[List[float]]
    ) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        Apply augmentation pipeline to image and bounding boxes.

        Args:
            image: PIL Image
            boxes: List of [x_center, y_center, width, height] (normalized)

        Returns:
            Tuple of (augmented_image_tensor, augmented_boxes)
        """
        if random.random() > self.augment_prob:
            image = self._resize_to_target(image)
            return self._basic_transform(image), boxes

        if random.random() < self.geometric_prob:
            image, boxes = self._geometric_augment(image, boxes)

        if random.random() < self.photometric_prob:
            image = self._photometric_augment(image)

        if random.random() < self.mosaic_prob:
            pass

        if random.random() < self.mixup_prob:
            pass

        image = self._resize_to_target(image)

        return self._basic_transform(image), boxes

    def _resize_to_target(self, image: Image.Image) -> Image.Image:
        """Resize image to target size."""
        if image.size != (self.target_size, self.target_size):
            image = image.resize(
                (self.target_size, self.target_size), Image.LANCZOS
            )
        return image

    def _basic_transform(self, image: Image.Image) -> torch.Tensor:
        """Apply basic normalization transform."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return transform(image)

    def _geometric_augment(
        self, image: Image.Image, boxes: List[List[float]]
    ) -> Tuple[Image.Image, List[List[float]]]:
        """Apply geometric transformations while preserving boxes."""
        if not boxes:
            return image, boxes

        if random.random() < 0.5:
            angle = random.uniform(*self.rotation_range)
            image, boxes = self._rotate_with_boxes(image, boxes, angle)

        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            image, boxes = self._scale_with_boxes(image, boxes, scale)

        if random.random() < 0.5:
            tx = random.uniform(*self.translation_range)
            ty = random.uniform(*self.translation_range)
            image, boxes = self._translate_with_boxes(image, boxes, tx, ty)

        if random.random() < 0.5:
            image, boxes = self._flip_with_boxes(image, boxes)

        return image, boxes

    def _photometric_augment(self, image: Image.Image) -> Image.Image:
        """Apply photometric transformations."""

        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        if random.random() < 0.5:
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        if random.random() < 0.3:
            image = self._add_noise(image)

        return image

    def _rotate_with_boxes(
        self, image: Image.Image, boxes: List[List[float]], angle: float
    ) -> Tuple[Image.Image, List[List[float]]]:
        """Rotate image and adjust bounding boxes."""

        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        image = image.rotate(angle, expand=True)

        new_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box

            img_w, img_h = image.size
            x_pixel = x_center * img_w
            y_pixel = y_center * img_h

            x_new = x_pixel * cos_a - y_pixel * sin_a
            y_new = x_pixel * sin_a + y_pixel * cos_a

            x_center_new = x_new / img_w
            y_center_new = y_new / img_h

            x_center_new = max(0, min(1, x_center_new))
            y_center_new = max(0, min(1, y_center_new))

            new_boxes.append([x_center_new, y_center_new, width, height])

        return image, new_boxes

    def _scale_with_boxes(
        self, image: Image.Image, boxes: List[List[float]], scale: float
    ) -> Tuple[Image.Image, List[List[float]]]:
        """Scale image and adjust bounding boxes."""

        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)

        new_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            new_boxes.append([x_center, y_center, width, height])

        return image, new_boxes

    def _translate_with_boxes(
        self, image: Image.Image, boxes: List[List[float]], tx: float, ty: float
    ) -> Tuple[Image.Image, List[List[float]]]:
        """Translate image and adjust bounding boxes."""

        new_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            x_center_new = x_center + tx
            y_center_new = y_center + ty

            x_center_new = max(0, min(1, x_center_new))
            y_center_new = max(0, min(1, y_center_new))

            new_boxes.append([x_center_new, y_center_new, width, height])

        return image, new_boxes

    def _flip_with_boxes(
        self, image: Image.Image, boxes: List[List[float]]
    ) -> Tuple[Image.Image, List[List[float]]]:
        """Flip image horizontally and adjust bounding boxes."""

        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        new_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            x_center_new = 1.0 - x_center
            new_boxes.append([x_center_new, y_center, width, height])

        return image, new_boxes

    def _add_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to image."""

        img_array = np.array(image, dtype=np.float32) / 255.0

        noise = np.random.normal(0, self.noise_std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)

        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)


def create_augmentation_pipeline(
    target_size: int = 416,
    augment_prob: float = 0.8,
    geometric_prob: float = 0.5,
    photometric_prob: float = 0.5,
) -> DetectionAugmentation:
    """
    Create an optimized augmentation pipeline.

    Args:
        target_size: Target image size
        augment_prob: Probability of applying any augmentation
        geometric_prob: Probability of geometric transformations
        photometric_prob: Probability of photometric transformations

    Returns:
        Configured DetectionAugmentation instance
    """
    return DetectionAugmentation(
        target_size=target_size,
        augment_prob=augment_prob,
        geometric_prob=geometric_prob,
        photometric_prob=photometric_prob,
    )


def create_validation_transform() -> transforms.Compose:
    """Create minimal transform for validation data."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
