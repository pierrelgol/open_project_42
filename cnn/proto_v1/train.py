"""
Training Script for CNN Object Detection

This script provides training with:
- Mixed precision training for faster training and lower memory usage
- Data augmentation pipeline
- Optimized DataLoader configuration
- Learning rate scheduling
- Monitoring and logging
"""

import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import time
import math
import argparse

from data.loader import COCO128Dataset
from data.augmentation import create_augmentation_pipeline
from model import create_model
from loss import get_loss


def collate_fn(batch):
    """Custom collate function for variable-length bounding boxes."""
    images = []
    targets = []
    for image, boxes in batch:
        images.append(image)
        targets.append(boxes)
    images = torch.stack(images)
    return images, targets


def encode_targets(targets, grid_h, grid_w):
    """
    Encode bounding boxes into (5, grid_h, grid_w) target tensor.
    Each cell: [x_rel, y_rel, w, h, conf], where x_rel, y_rel are relative to
    grid cell center. Assumes normalized coordinates (YOLO format).
    """
    batch_size = len(targets)
    target_tensor = torch.zeros((batch_size, 5, grid_h, grid_w))

    for b, boxes in enumerate(targets):
        for box in boxes:
            x_center, y_center, w, h = box

            grid_x = int(x_center * grid_w)
            grid_y = int(y_center * grid_h)

            if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                x_rel = x_center * grid_w - grid_x
                y_rel = y_center * grid_h - grid_y

                target_tensor[b, 0, grid_y, grid_x] = x_rel
                target_tensor[b, 1, grid_y, grid_x] = y_rel
                target_tensor[b, 2, grid_y, grid_x] = w
                target_tensor[b, 3, grid_y, grid_x] = h
                target_tensor[b, 4, grid_y, grid_x] = 1.0

    return target_tensor


def create_dataloaders(
    data_dir, batch_size, val_split, num_workers, use_augmentation=True
):
    """Create training and validation dataloaders."""
    print("Creating dataloaders...")

    if use_augmentation:
        train_transform = create_augmentation_pipeline(
            target_size=416,
            augment_prob=0.8,
            geometric_prob=0.5,
            photometric_prob=0.5,
        )
    else:
        train_transform = None

    full_dataset = COCO128Dataset(data_dir=data_dir, transform=train_transform)

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers and num_workers > 0 else False,
        prefetch_factor=2 if num_workers and num_workers > 0 else 2,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers and num_workers > 0 else False,
        prefetch_factor=2 if num_workers and num_workers > 0 else 2,
        drop_last=False,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def create_optimizer_and_scheduler(
    model, learning_rate, num_epochs, steps_per_epoch
):
    """Create optimizer and learning rate scheduler."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def get_lr_scheduler(optimizer, num_epochs, steps_per_epoch):
        warmup_epochs = max(1, int(0.1 * num_epochs))
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_lr_scheduler(optimizer, num_epochs, steps_per_epoch)
    return optimizer, scheduler


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    num_epochs,
    print_interval,
    use_mixed_precision=True,
):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (images, targets) in enumerate(train_loader, 1):
        images = images.to(device)
        target_tensor = encode_targets(targets, model.grid_h, model.grid_w).to(
            device
        )

        optimizer.zero_grad()

        if use_mixed_precision:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, target_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()

        scheduler.step()
        running_loss += loss.item()

        if batch_idx % print_interval == 0:
            avg_loss = running_loss / print_interval
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{num_batches}] "
                f"Loss: {avg_loss:.4f} LR: {current_lr:.2e}"
            )

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    sample_pred = outputs[0]
                    max_conf = sample_pred[4].max().item()
                    mean_conf = sample_pred[4].mean().item()
                    print(
                        f"  Max confidence: {max_conf:.3f}, Mean: {mean_conf:.3f}"
                    )

            running_loss = 0.0

    return running_loss / max(1, num_batches)


def validate_epoch(
    model, val_loader, criterion, device, use_mixed_precision=True
):
    """Validate for one epoch."""
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            target_tensor = encode_targets(
                targets, model.grid_h, model.grid_w
            ).to(device)

            if use_mixed_precision:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, target_tensor)
            else:
                outputs = model(images)
                loss = criterion(outputs, target_tensor)

            val_loss += loss.item()
            val_batches += 1

    return val_loss / max(1, val_batches)


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN Object Detection Model"
    )
    parser.add_argument(
        "--data-dir", default="data/coco128", help="Dataset directory"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of workers"
    )
    parser.add_argument(
        "--print-interval", type=int, default=10, help="Print interval"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save interval"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision",
    )
    parser.add_argument(
        "--no-augmentation", action="store_true", help="Disable augmentation"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.num_workers is None:
        args.num_workers = min(8, os.cpu_count() or 1)
        print(f"Auto-detected {args.num_workers} workers")

    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        use_augmentation=not args.no_augmentation,
    )

    model = create_model(input_size=416).to(device)
    criterion = get_loss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, args.learning_rate, args.epochs, len(train_loader)
    )

    use_mixed_precision = not args.no_mixed_precision and device.type == "cuda"
    scaler = GradScaler() if use_mixed_precision else None

    if use_mixed_precision:
        print("Using mixed precision training")
    else:
        print("Using full precision training")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            args.epochs,
            args.print_interval,
            use_mixed_precision,
        )

        val_loss = validate_epoch(
            model, val_loader, criterion, device, use_mixed_precision
        )

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch}/{args.epochs}] completed in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"model_epoch_{epoch}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    final_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
