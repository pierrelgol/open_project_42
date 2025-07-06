"""
COCO128 Dataset Loader

This module provides a PyTorch Dataset class for loading COCO128 images and
bounding boxes with transforms for object detection tasks.
"""

import sys
import os
import multiprocessing
import inspect
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import resize_image_square


class COCO128Dataset(Dataset):
    """
    PyTorch Dataset for COCO128 with YOLO format labels.

    Returns:
        - image: torch.Tensor of shape (1, target_size, target_size) - grayscale, normalized
        - targets: List of [x_center, y_center, width, height] bounding boxes

    """

    def __init__(
        self,
        data_dir: str = "coco128",
        transform: Optional[transforms.Compose] = None,
        target_size: int = 416,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to COCO128 dataset directory
            transform: Optional transforms to apply to images
            target_size: Target size for resizing images (default 416)
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size

        images_dir = self.data_dir / "images"
        if not images_dir.exists():
            images_dir = self.data_dir / "train2017"
            if not images_dir.exists():
                raise FileNotFoundError(
                    f"Images directory not found in {self.data_dir}. "
                    f"Expected 'images' or 'train2017' subdirectory."
                )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f
            for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        if len(image_files) == 0:
            train2017_dir = images_dir / "train2017"
            if train2017_dir.exists():
                images_dir = train2017_dir
                image_files = [
                    f
                    for f in images_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
        if len(image_files) == 0:
            raise FileNotFoundError(
                f"No image files found in {images_dir}. "
                f"Supported formats: {image_extensions}"
            )

        labels_dir = self.data_dir / "labels"
        if not labels_dir.exists():
            labels_dir = self.data_dir / "labels" / "train2017"
            if not labels_dir.exists():
                raise FileNotFoundError(
                    f"Labels directory not found in {self.data_dir}. "
                    f"Expected 'labels' or 'labels/train2017' subdirectory."
                )

        label_files = list(labels_dir.iterdir()) if labels_dir.exists() else []
        if len(label_files) == 0:
            train2017_labels_dir = labels_dir / "train2017"
            if train2017_labels_dir.exists():
                labels_dir = train2017_labels_dir

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = image_files
        self.image_files.sort()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transform

        print(f"Loaded {len(self.image_files)} images from {self.data_dir}")
        print(f"Target size: {self.target_size}x{self.target_size}")
        print(f"Output format: grayscale (1 channel)")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        Get a single image and its bounding boxes.

        Args:
            idx: Index of the image to load

        Returns:
            Tuple of (image_tensor, bounding_boxes)
            - image_tensor: torch.Tensor of shape (1, target_size, target_size)
            - bounding_boxes: List of [x_center, y_center, width, height] (normalized YOLO format)
        """

        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("L")

        resized_image = resize_image_square(image, self.target_size)

        label_filename = f"{image_path.stem}.txt"
        label_path = self.labels_dir / label_filename

        if not label_path.exists():
            train2017_labels_dir = self.labels_dir / "train2017"
            if train2017_labels_dir.exists():
                label_path = train2017_labels_dir / label_filename

        bounding_boxes = self._load_labels(label_path)

        if hasattr(self.transform, "__call__"):
            sig = inspect.signature(self.transform.__call__)
            if len(sig.parameters) >= 2:
                image_tensor, bounding_boxes = self.transform(
                    resized_image, bounding_boxes
                )
            else:
                image_tensor = self.transform(resized_image)
        else:
            image_tensor = self.transform(resized_image)

        if idx < 3:
            print(f"Image {idx}: {image_path.name}")
            print(f"  Original size: {image.size}")
            print(f"  Resized size: {resized_image.size}")
            print(f"  Tensor shape: {image_tensor.shape}")
            print(
                f"  Expected shape: (1, {self.target_size}, {self.target_size})"
            )

        return image_tensor, bounding_boxes

    def _load_labels(self, label_path: Path) -> List[List[float]]:
        """
        Load YOLO format labels and keep coordinates as-is (normalized).

        Args:
            label_path: Path to the label file

        Returns:
            List of [x_center, y_center, width, height] bounding boxes (normalized YOLO format)
        """
        bounding_boxes = []

        if not label_path.exists():
            return bounding_boxes

        try:
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            if (
                                0 <= x_center <= 1
                                and 0 <= y_center <= 1
                                and 0 <= width <= 1
                                and 0 <= height <= 1
                            ):
                                bounding_boxes.append([
                                    x_center,
                                    y_center,
                                    width,
                                    height,
                                ])
                            else:
                                print(
                                    f"Warning: Invalid coordinates in {label_path}: "
                                    f"{x_center}, {y_center}, {width}, {height}"
                                )

        except Exception as e:
            print(f"Error loading labels from {label_path}: {e}")

        return bounding_boxes

    def get_image_info(self, idx: int) -> dict:
        """
        Get detailed information about a specific image in the dataset.

        Args:
            idx: Index of the image

        Returns:
            Dictionary containing image information:
            - 'path': Path to the image file
            - 'size': Original image size (width, height)
            - 'target_size': Resized target size
            - 'num_boxes': Number of bounding boxes
            - 'boxes': List of bounding boxes in YOLO format
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        image_path = self.image_files[idx]
        image = Image.open(image_path)

        label_filename = f"{image_path.stem}.txt"
        label_path = self.labels_dir / label_filename

        if not label_path.exists():
            train2017_labels_dir = self.labels_dir / "train2017"
            if train2017_labels_dir.exists():
                label_path = train2017_labels_dir / label_filename

        bounding_boxes = self._load_labels(label_path)

        return {
            "path": str(image_path),
            "size": image.size,
            "target_size": self.target_size,
            "num_boxes": len(bounding_boxes),
            "boxes": bounding_boxes,
        }


def load_dataset(
    data_dir: str = "coco128",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader for the COCO128 dataset.

    Args:
        data_dir: Path to the extracted COCO128 dataset
        batch_size: Batch size for the DataLoader (optimized for GPU memory)
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for loading (auto-detect if None)
        persistent_workers: Keep workers alive between epochs for faster startup
        pin_memory: Enable memory pinning for faster GPU transfers
        prefetch_factor: Number of batches to prefetch per worker
        drop_last: Whether to drop incomplete batches

    Returns:
        Optimized PyTorch DataLoader for the COCO128 dataset
    """
    dataset = COCO128Dataset(data_dir=data_dir)

    if num_workers is None:
        num_workers = min(8, multiprocessing.cpu_count())

    num_workers = max(1, min(num_workers, len(dataset)))

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 8:
            batch_size = min(batch_size, 32)
        elif gpu_memory >= 4:
            batch_size = min(batch_size, 16)
        else:
            batch_size = min(batch_size, 8)

    def collate_fn(batch):
        images = []
        targets = []

        for image, boxes in batch:
            images.append(image)
            targets.append(boxes)

        images = torch.stack(images)
        return images, targets

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=drop_last,
        generator=torch.Generator(device="cpu") if shuffle else None,
    )

    print(f"Optimized DataLoader created:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")
    print(f"  - Persistent workers: {persistent_workers and num_workers > 0}")
    print(f"  - Pin memory: {pin_memory and torch.cuda.is_available()}")
    print(f"  - Prefetch factor: {prefetch_factor if num_workers > 0 else 2}")
    print(f"  - Total batches: {len(dataloader)}")

    return dataloader


if __name__ == "__main__":
    try:
        dataset = COCO128Dataset()
        print(f"Dataset loaded successfully: {len(dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "Make sure to run extract_coco128.py first to extract the dataset."
        )
