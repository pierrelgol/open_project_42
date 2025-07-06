import torch
import torch.nn as nn
import torchvision.models as models
from utils import yolo_to_pixel_coords, nms


class ModelCNN(nn.Module):
    """
    CNN model using ResNet18 backbone for grayscale input with detection head.
    
    Output: (batch, 5, grid_h, grid_w) where 5 = [x, y, w, h, confidence]
    Grid size: calculated from input size (32x downsampling)
    """

    def __init__(self, num_classes=None, input_size=416):
        super(ModelCNN, self).__init__()

        self.grid_h = input_size // 32
        self.grid_w = input_size // 32
        self.input_size = input_size

        resnet18 = models.resnet18(pretrained=True)

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
        )

        backbone_output_size = input_size // 32

        if (
            backbone_output_size != self.grid_h
            or backbone_output_size != self.grid_w
        ):
            stride_h = backbone_output_size // self.grid_h
            stride_w = backbone_output_size // self.grid_w
            stride = max(stride_h, stride_w)

            self.detection_head = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 5, kernel_size=1, stride=stride, padding=0),
            )
        else:
            self.detection_head = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 5, kernel_size=1, stride=1, padding=0),
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for detection head layers."""
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 5:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor (batch, 1, H, W) - grayscale image

        Returns:
            Output tensor (batch, 5, grid_h, grid_w) where:
            - 5 channels: [x, y, w, h, confidence]
            - grid_h x grid_w spatial dimensions
        """

        features = self.backbone(x)

        output = self.detection_head(features)

        output = torch.sigmoid(output)

        return output

    def get_output_shape(self, input_shape):
        """
        Get output shape for given input shape.

        Args:
            input_shape: Tuple of (batch, channels, height, width)

        Returns:
            Tuple of output shape (batch, 5, grid_h, grid_w)
        """
        batch_size = input_shape[0]
        return (batch_size, 5, self.grid_h, self.grid_w)

    def inference(
        self,
        image,
        conf_thresh=0.5,
        iou_thresh=0.5,
        img_size=None,
        max_detections=5,
    ):
        """
        Run inference on a single image tensor.
        Args:
            image: torch.Tensor (1, H, W) or (1, 1, H, W) - resized image
            conf_thresh: confidence threshold
            iou_thresh: IoU threshold for NMS
            img_size: (width, height) of original image for visualization
            max_detections: max number of detections to return
        Returns:
            List of detections: [x1, y1, x2, y2, confidence] in original coordinates
        """
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)

            output = self.forward(image)
            output = output[0]

            if img_size is None:
                img_size = (image.shape[3], image.shape[2])

            boxes = []
            scores = []
            S_h, S_w = output.shape[1], output.shape[2]

            for i in range(S_h):
                for j in range(S_w):
                    conf = output[4, i, j].item()
                    if conf >= conf_thresh:
                        x_rel = output[0, i, j].item()
                        y_rel = output[1, i, j].item()
                        w = output[2, i, j].item()
                        h = output[3, i, j].item()

                        x_abs = (j + x_rel) / S_w
                        y_abs = (i + y_rel) / S_h

                        x_abs = max(0.0, min(1.0, x_abs))
                        y_abs = max(0.0, min(1.0, y_abs))
                        w = max(0.01, min(1.0, w))
                        h = max(0.01, min(1.0, h))

                        x1, y1, x2, y2 = yolo_to_pixel_coords(
                            [x_abs, y_abs, w, h], img_size
                        )

                        x1 = max(0, min(img_size[0] - 1, x1))
                        y1 = max(0, min(img_size[1] - 1, y1))
                        x2 = max(x1 + 1, min(img_size[0], x2))
                        y2 = max(y1 + 1, min(img_size[1], y2))

                        boxes.append([x1, y1, x2, y2])
                        scores.append(conf)

            keep = nms(boxes, scores, iou_threshold=iou_thresh)
            detections = [[*boxes[idx], scores[idx]] for idx in keep]

            detections.sort(key=lambda x: x[4], reverse=True)
            detections = detections[:max_detections]

        return detections


def create_model(num_classes=None, input_size=416):
    """
    Factory function to create CNN model.

    Args:
        num_classes: Not used in this implementation (kept for compatibility)
        input_size: Input size for the model (default 416). Must be divisible by 32.

    Returns:
        ModelCNN instance with ResNet18 backbone and custom detection head

    Note:
        The model expects grayscale input images and outputs YOLO-style predictions
        with 5 channels per grid cell: [x_rel, y_rel, w, h, confidence].
        Grid size is automatically calculated as input_size // 32.
    """
    return ModelCNN(num_classes=num_classes, input_size=input_size)


if __name__ == "__main__":
    model = create_model(input_size=416)
    print(
        f"Model created successfully with grid size: {model.grid_h}x{model.grid_w}"
    )
