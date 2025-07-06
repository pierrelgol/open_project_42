"""
Main entry point for the CNN object detection model.

This script provides a command-line interface for:
- inference: Run inference on a single image
- visualize: Show predictions on sample images
"""

import argparse
import os
import traceback

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from data.loader import COCO128Dataset
from model import create_model
from utils import (
    load_checkpoint,
    yolo_to_pixel_coords,
    resize_image_square,
)


def run_inference(args):
    """Run inference on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(input_size=416).to(device)
    if args.model_path and os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path, device)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Warning: No model checkpoint provided, using untrained model")

    try:
        image = Image.open(args.image_path).convert("L")
        original_size = image.size
        print(f"Original image size: {original_size}")

        resized_image = resize_image_square(image, target_size=416)
        print(f"Resized to: {resized_image.size}")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(resized_image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            detections = model.inference(
                image_tensor[0],
                conf_thresh=args.conf_threshold,
                iou_thresh=args.iou_threshold,
                img_size=(416, 416),
            )

        print(f"Found {len(detections)} detections")

        if args.output_path:
            vis_image = resized_image.convert("RGB")
            draw = ImageDraw.Draw(vis_image)

            for x1, y1, x2, y2, conf in detections:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 10), f"{conf:.2f}", fill="red")

            vis_image.save(args.output_path)
            print(f"Output saved to: {args.output_path}")

        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            print(
                f"Detection {i}: bbox=({x1}, {y1}, {x2}, {y2}), conf={conf:.3f}"
            )

    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc()
        return False

    return True


def visualize_predictions(args):
    """Visualize predictions on sample images from the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(input_size=416).to(device)
    if args.model_path and os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path, device)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Warning: No model checkpoint provided, using untrained model")

    dataset = COCO128Dataset(data_dir=args.data_dir, target_size=416)
    print(f"Dataset size: {len(dataset)}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    num_samples = min(5, len(dataset))
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    for i in range(num_samples):
        image, targets = dataset[i]
        original_size = (416, 416)

        model.eval()
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            detections = model.inference(
                image_tensor[0],
                conf_thresh=args.conf_threshold,
                iou_thresh=args.iou_threshold,
                img_size=original_size,
            )

        image_pil = transforms.ToPILImage()(image).convert("RGB")
        draw = ImageDraw.Draw(image_pil)

        for box in targets:
            x1, y1, x2, y2 = yolo_to_pixel_coords(box, original_size)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

        for x1, y1, x2, y2, conf in detections:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), f"{conf:.2f}", fill="red")

        axes[0, i].imshow(image_pil)
        axes[0, i].set_title(f"Sample {i + 1}")
        axes[0, i].axis("off")

        gt_image = transforms.ToPILImage()(image).convert("RGB")
        gt_draw = ImageDraw.Draw(gt_image)
        for box in targets:
            x1, y1, x2, y2 = yolo_to_pixel_coords(box, original_size)
            gt_draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

        axes[1, i].imshow(gt_image)
        axes[1, i].set_title(f"Ground Truth {i + 1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {args.output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="CNN Object Detection Model - Inference and Visualization"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Available modes")

    inference_parser = subparsers.add_parser(
        "inference", help="Run inference on image"
    )
    inference_parser.add_argument("image_path", help="Path to input image")
    inference_parser.add_argument(
        "--model-path", help="Path to trained model checkpoint"
    )
    inference_parser.add_argument(
        "--output-path", help="Path to save output image with detections"
    )
    inference_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )
    inference_parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold for NMS"
    )

    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize predictions"
    )
    visualize_parser.add_argument(
        "--data-dir",
        default="data/coco128",
        help="Path to COCO128 dataset directory",
    )
    visualize_parser.add_argument(
        "--model-path", help="Path to trained model checkpoint"
    )
    visualize_parser.add_argument(
        "--output-path", help="Path to save visualization image"
    )
    visualize_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )
    visualize_parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold for NMS"
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    if args.mode == "inference":
        run_inference(args)
    elif args.mode == "visualize":
        visualize_predictions(args)


if __name__ == "__main__":
    main()
