from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Segmenter:
    """
    SAM2-based segmentation for objects in images.

    This class wraps SAM2 to provide easy-to-use segmentation functionality
    for objects specified via bounding boxes or point prompts.
    """

    def __init__(
        self,
        checkpoint_path: str = "../checkpoints/sam2.1_hiera_large.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize SAM2 segmenter.

        Args:
            checkpoint_path: Path to SAM2 checkpoint file
            model_cfg: Path to model configuration file
            device: Device to run inference on (cuda/mps/cpu). Auto-detected if None.
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        print(f"[SAM2Segmenter] Using device: {device}")

        # Enable optimizations for CUDA
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 model
        self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        print(f"[SAM2Segmenter] Model loaded from {checkpoint_path}")

    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        """
        Set the image for segmentation.

        Args:
            image: Input image as numpy array (H, W, 3) or PIL Image
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        self.predictor.set_image(image)
        self.current_image = image

    def segment_from_box(
        self,
        box: Union[List[float], np.ndarray],
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment object using bounding box prompt.

        Args:
            box: Bounding box in xyxy format [x1, y1, x2, y2]
            multimask_output: Whether to output multiple mask candidates

        Returns:
            masks: Binary masks (N, H, W)
            scores: Quality scores for each mask
            logits: Low-resolution mask logits
        """
        box = np.array(box)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=multimask_output,
        )
        return masks, scores, logits

    def segment_from_points(
        self,
        points: Union[List[List[float]], np.ndarray],
        labels: Union[List[int], np.ndarray],
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment object using point prompts.

        Args:
            points: Point coordinates (N, 2) in [x, y] format
            labels: Point labels (1 = foreground, 0 = background)
            multimask_output: Whether to output multiple mask candidates

        Returns:
            masks: Binary masks (N, H, W)
            scores: Quality scores for each mask
            logits: Low-resolution mask logits
        """
        points = np.array(points)
        labels = np.array(labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )

        # Sort by score (best first)
        if multimask_output:
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

        return masks, scores, logits

    def segment_from_box_and_points(
        self,
        box: Union[List[float], np.ndarray],
        points: Optional[Union[List[List[float]], np.ndarray]] = None,
        labels: Optional[Union[List[int], np.ndarray]] = None,
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment object using both bounding box and point prompts.

        Args:
            box: Bounding box in xyxy format [x1, y1, x2, y2]
            points: Point coordinates (N, 2) in [x, y] format
            labels: Point labels (1 = foreground, 0 = background)
            multimask_output: Whether to output multiple mask candidates

        Returns:
            masks: Binary masks (N, H, W)
            scores: Quality scores for each mask
            logits: Low-resolution mask logits
        """
        box = np.array(box)
        points = np.array(points) if points is not None else None
        labels = np.array(labels) if labels is not None else None

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=multimask_output,
        )
        return masks, scores, logits

    def segment_batch_boxes(
        self,
        boxes: Union[List[List[float]], np.ndarray],
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment multiple objects using multiple bounding box prompts.

        Args:
            boxes: Multiple bounding boxes (N, 4) in xyxy format
            multimask_output: Whether to output multiple mask candidates per box

        Returns:
            masks: Binary masks (N, M, H, W) where N=num_boxes, M=num_masks
            scores: Quality scores for each mask
            logits: Low-resolution mask logits
        """
        boxes = np.array(boxes)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=multimask_output,
        )
        return masks, scores, logits

    def get_best_mask(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Select the best mask from multiple candidates.

        Args:
            masks: Multiple masks (N, H, W)
            scores: Quality scores for each mask

        Returns:
            Best mask (H, W)
        """
        best_idx = np.argmax(scores)
        return masks[best_idx]


def segment_objects_from_instruction(
    image: Union[np.ndarray, Image.Image],
    object_positions: Dict[str, np.ndarray],
    segmenter: SAM2Segmenter,
    use_box_prompt: bool = True,
    box_margin: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Segment objects mentioned in instructions based on their positions.

    Args:
        image: Input image as numpy array (H, W, 3) or PIL Image
        object_positions: Dictionary mapping object names to 3D positions (x, y, z)
        segmenter: Initialized SAM2Segmenter instance
        use_box_prompt: Whether to use bounding boxes (True) or points (False)
        box_margin: Margin around estimated object position for box prompt

    Returns:
        Dictionary mapping object names to segmentation masks (H, W)
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    segmenter.set_image(image)

    masks_dict = {}
    h, w = image.shape[:2]

    for obj_name, position_3d in object_positions.items():
        # Convert 3D position to approximate 2D position
        # This is a simplified projection - adjust based on your camera model
        x_2d = int((position_3d[0] / 2.0 + 0.5) * w)  # Normalize to image coords
        y_2d = int((position_3d[1] / 2.0 + 0.5) * h)

        # Clamp to image bounds
        x_2d = np.clip(x_2d, 0, w - 1)
        y_2d = np.clip(y_2d, 0, h - 1)

        if use_box_prompt:
            # Create bounding box around estimated position
            box_w = int(w * box_margin)
            box_h = int(h * box_margin)
            box = [
                max(0, x_2d - box_w // 2),
                max(0, y_2d - box_h // 2),
                min(w, x_2d + box_w // 2),
                min(h, y_2d + box_h // 2),
            ]
            masks, scores, _ = segmenter.segment_from_box(box, multimask_output=False)
        else:
            # Use point prompt
            points = [[x_2d, y_2d]]
            labels = [1]  # Foreground
            masks, scores, _ = segmenter.segment_from_points(
                points, labels, multimask_output=True
            )

        # Get best mask
        best_mask = segmenter.get_best_mask(masks, scores)
        masks_dict[obj_name] = best_mask

        print(
            f"[SAM2Segmenter] Segmented '{obj_name}' at position ({x_2d}, {y_2d}), "
            f"mask score: {scores.max():.3f}"
        )

    return masks_dict


def visualize_masks(
    image: np.ndarray,
    masks_dict: Dict[str, np.ndarray],
    alpha: float = 0.6,
    show: bool = True,
) -> np.ndarray:
    """
    Visualize segmentation masks overlaid on image.

    Args:
        image: Original image (H, W, 3)
        masks_dict: Dictionary mapping object names to masks
        alpha: Transparency of mask overlay
        show: Whether to display the image

    Returns:
        Image with masks overlaid
    """
    import matplotlib.pyplot as plt

    overlay = image.copy()
    np.random.seed(42)

    for obj_name, mask in masks_dict.items():
        # Generate random color for each object
        color = np.random.rand(3) * 255
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color

        # Blend with original image
        overlay = overlay * (1 - alpha * mask[..., None]) + mask_colored * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(overlay)
        plt.title("Segmented Objects")
        plt.axis("off")

        # Add legend
        legend_elements = []
        for obj_name in masks_dict.keys():
            from matplotlib.patches import Patch

            color = np.random.rand(3)
            legend_elements.append(Patch(facecolor=color, label=obj_name))
        plt.legend(handles=legend_elements, loc="upper right")
        plt.show()

    return overlay
