"""
COCO128 Dataset Extractor

This script extracts the COCO128 dataset from coco128.zip and organizes it into
the proper directory structure with images and YOLO-format labels.
"""

import zipfile
import shutil
from pathlib import Path
from typing import Tuple
import argparse


class COCO128Extractor:
    def __init__(
        self,
        zip_path: str = "data/coco128.zip",
        output_dir: str = "data/coco128",
    ):
        """
        Initialize the COCO128 extractor.

        Args:
            zip_path: Path to the coco128.zip file
            output_dir: Directory to extract the dataset to
        """
        self.zip_path = Path(zip_path)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

    def check_already_extracted(self) -> bool:
        """
        Check if the dataset is already extracted.

        Returns:
            True if already extracted, False otherwise
        """
        if not self.output_dir.exists():
            return False

        if not (self.images_dir.exists() and self.labels_dir.exists()):
            return False

        image_files = list(self.images_dir.glob("*.jpg")) + list(
            self.images_dir.glob("*.png")
        )
        label_files = list(self.labels_dir.glob("*.txt"))

        return len(image_files) > 0 and len(label_files) > 0

    def create_directories(self):
        """Create the necessary directories for the dataset."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directories: {self.images_dir}, {self.labels_dir}")

    def extract_zip(self):
        """Extract the coco128.zip file to a temporary location."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")

        print(f"Extracting {self.zip_path}...")

        temp_dir = self.output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        print("Extraction completed.")
        return temp_dir

    def find_coco_files(self, temp_dir: Path) -> Tuple[Path, Path]:
        """
        Find the images and labels directories in the extracted content.

        Args:
            temp_dir: Temporary directory containing extracted files

        Returns:
            Tuple of (images_directory, labels_directory)
        """

        images_dir = None
        labels_dir = None

        for item in temp_dir.iterdir():
            if item.is_dir():
                if (item / "images").exists():
                    images_subdir = item / "images"
                    for subdir in images_subdir.iterdir():
                        if subdir.is_dir():
                            image_files = list(subdir.glob("*.jpg")) + list(
                                subdir.glob("*.png")
                            )
                            if image_files:
                                images_dir = subdir
                                break

                if (item / "labels").exists():
                    labels_subdir = item / "labels"
                    for subdir in labels_subdir.iterdir():
                        if subdir.is_dir():
                            label_files = list(subdir.glob("*.txt"))
                            if label_files:
                                labels_dir = subdir
                                break

        if not images_dir:
            raise FileNotFoundError("No images directory found in the zip")

        if not labels_dir:
            raise FileNotFoundError("No labels directory found in the zip")

        print(f"Found images directory: {images_dir}")
        print(f"Found labels directory: {labels_dir}")

        return images_dir, labels_dir

    def copy_yolo_files(self, images_dir: Path, labels_dir: Path):
        """
        Copy pre-processed YOLO format files to output directory.

        Args:
            images_dir: Path to directory containing images
            labels_dir: Path to directory containing YOLO labels
        """
        print("Copying pre-processed YOLO format files...")

        image_files = list(images_dir.glob("*.jpg")) + list(
            images_dir.glob("*.png")
        )
        print(f"Found {len(image_files)} image files")

        for image_path in image_files:
            output_image_path = self.images_dir / image_path.name
            shutil.copy2(image_path, output_image_path)

        label_files = list(labels_dir.glob("*.txt"))
        print(f"Found {len(label_files)} label files")

        for label_path in label_files:
            output_label_path = self.labels_dir / label_path.name
            shutil.copy2(label_path, output_label_path)

        print(
            f"Successfully copied {len(image_files)} images and {len(label_files)} labels"
        )

    def cleanup_temp(self, temp_dir: Path):
        """Remove temporary directory."""
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files")

    def extract(self, force: bool = False):
        """
        Extract and process the COCO128 dataset.

        Args:
            force: If True, re-extract even if already extracted
        """
        print("COCO128 Dataset Extractor")
        print("=" * 40)

        if not force and self.check_already_extracted():
            print(f"Dataset already extracted to {self.output_dir}")
            print("Use --force to re-extract")
            return

        try:
            self.create_directories()

            temp_dir = self.extract_zip()

            images_dir, labels_dir = self.find_coco_files(temp_dir)

            self.copy_yolo_files(images_dir, labels_dir)

            self.cleanup_temp(temp_dir)

            print("\nExtraction completed successfully!")
            print(f"Dataset location: {self.output_dir}")
            print(f"Images: {self.images_dir}")
            print(f"Labels: {self.labels_dir}")

            image_count = len(list(self.images_dir.glob("*.jpg"))) + len(
                list(self.images_dir.glob("*.png"))
            )
            label_count = len(list(self.labels_dir.glob("*.txt")))
            print(f"Total images: {image_count}")
            print(f"Total label files: {label_count}")

        except Exception as e:
            print(f"Error during extraction: {e}")
            raise


def main():
    """Main function to run the extractor."""
    parser = argparse.ArgumentParser(description="Extract COCO128 dataset")
    parser.add_argument(
        "--zip-path",
        default="data/coco128.zip",
        help="Path to coco128.zip file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/coco128",
        help="Output directory for extracted dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if already extracted",
    )

    args = parser.parse_args()

    extractor = COCO128Extractor(args.zip_path, args.output_dir)
    extractor.extract(force=args.force)


def extract_dataset(data_dir="coco128"):
    """
    Ensure the COCO128 dataset is present in the specified directory.
    If already present, do nothing. If not, attempt extraction if zip exists.
    """
    extractor = COCO128Extractor(
        zip_path=f"{data_dir}.zip", output_dir=data_dir
    )
    if extractor.check_already_extracted():
        print(
            f"COCO128 dataset already present in {data_dir}. Skipping extraction."
        )
        return
    if extractor.zip_path.exists():
        extractor.extract(force=False)
    else:
        raise FileNotFoundError(
            f"COCO128 dataset not found in {data_dir} and zip file {extractor.zip_path} is missing."
        )


if __name__ == "__main__":
    main()
