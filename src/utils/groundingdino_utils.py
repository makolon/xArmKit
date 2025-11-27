from typing import Dict, List, Optional, Tuple, Union

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image, ImageDraw, ImageFont


class GroundingDINODetector:
    """
    GroundingDINO-based object detector for text-guided bounding box detection.

    This class wraps GroundingDINO to provide easy-to-use object detection
    functionality from natural language descriptions.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        Initialize GroundingDINO detector.

        Args:
            config_path: Path to GroundingDINO config file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on (cuda/cpu). Auto-detected if None.
            box_threshold: Confidence threshold for box detection
            text_threshold: Confidence threshold for text matching
        """

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(f"[GroundingDINO] Using device: {device}")

        # Load model configuration
        args = SLConfig.fromfile(config_path)
        args.device = device

        # Build model
        self.model = build_model(args)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        load_res = self.model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(f"[GroundingDINO] Model loaded from {checkpoint_path}")
        print(f"[GroundingDINO] Load result: {load_res}")

        self.model.eval()
        self.model = self.model.to(device)

        # Image transformation
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_image(
        self, image: Union[np.ndarray, Image.Image]
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Preprocess image for GroundingDINO.

        Args:
            image: Input image as numpy array or PIL Image

        Returns:
            Tuple of (original PIL image, preprocessed tensor)
        """
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image).convert("RGB")
        else:
            image_pil = image.convert("RGB")

        image_tensor, _ = self.transform(image_pil, None)
        return image_pil, image_tensor

    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        with_logits: bool = True,
    ) -> Tuple[torch.Tensor, List[str], List[float]]:
        """
        Detect objects in image based on text prompt.

        Args:
            image: Input image
            text_prompt: Text description of objects to detect (e.g., "banana . cup . plate")
            box_threshold: Box confidence threshold (uses default if None)
            text_threshold: Text matching threshold (uses default if None)
            with_logits: Whether to include confidence scores in phrases

        Returns:
            Tuple of (boxes, phrases, scores)
            - boxes: Tensor of shape (N, 4) in xyxy format, normalized to [0, 1]
            - phrases: List of detected object phrases
            - scores: List of confidence scores for each detection
        """
        # Use default thresholds if not specified
        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold

        # Preprocess image
        image_pil, image_tensor = self.preprocess_image(image)

        # Prepare caption
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        # Run inference
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            # Ensure float32 for CUDA compatibility with ms_deform_attn
            if self.device == "cuda":
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.model(image_tensor[None].float(), captions=[caption])
            else:
                outputs = self.model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # Filter output
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # Get phrases
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            )
            score = logit.max().item()
            scores.append(score)

            if with_logits:
                pred_phrases.append(pred_phrase + f"({score:.3f})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases, scores

    def detect_batch(
        self,
        image: Union[np.ndarray, Image.Image],
        object_names: List[str],
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict[str, Dict[str, Union[torch.Tensor, float]]]:
        """
        Detect multiple objects and return results organized by object name.

        Args:
            image: Input image
            object_names: List of object names to detect
            box_threshold: Box confidence threshold
            text_threshold: Text matching threshold

        Returns:
            Dictionary mapping object names to detection results:
            {
                "object_name": {
                    "boxes": tensor of shape (N, 4),
                    "scores": list of confidence scores,
                    "best_box": tensor of shape (4,) - highest confidence box,
                    "best_score": float - highest confidence score
                }
            }
        """
        # Create text prompt from object names
        text_prompt = " . ".join(object_names)

        # Detect all objects
        boxes, phrases, scores = self.detect(
            image, text_prompt, box_threshold, text_threshold, with_logits=False
        )

        # Organize results by object name
        results = {name: {"boxes": [], "scores": []} for name in object_names}

        for box, phrase, score in zip(boxes, phrases, scores):
            # Match phrase to object name (case-insensitive, partial match)
            phrase_lower = phrase.lower().strip()
            for obj_name in object_names:
                if obj_name.lower() in phrase_lower or phrase_lower in obj_name.lower():
                    results[obj_name]["boxes"].append(box)
                    results[obj_name]["scores"].append(score)
                    break

        # Convert to tensors and add best box/score
        for obj_name in object_names:
            if results[obj_name]["boxes"]:
                results[obj_name]["boxes"] = torch.stack(results[obj_name]["boxes"])
                best_idx = np.argmax(results[obj_name]["scores"])
                results[obj_name]["best_box"] = results[obj_name]["boxes"][best_idx]
                results[obj_name]["best_score"] = results[obj_name]["scores"][best_idx]
            else:
                results[obj_name]["boxes"] = torch.empty((0, 4))
                results[obj_name]["best_box"] = None
                results[obj_name]["best_score"] = 0.0

        return results

    def boxes_to_xyxy(
        self, boxes: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Convert normalized center format boxes to xyxy pixel coordinates.

        Args:
            boxes: Boxes in normalized cxcywh format (N, 4)
            image_size: (height, width) of image

        Returns:
            Boxes in xyxy pixel format (N, 4)
        """
        H, W = image_size

        # Convert from cxcywh to xyxy
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, :2] -= boxes[:, 2:] / 2  # top-left
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]  # bottom-right

        # Scale to pixel coordinates
        boxes_xyxy = boxes_xyxy * torch.tensor([W, H, W, H])

        return boxes_xyxy.long()


def visualize_detections(
    image: Union[np.ndarray, Image.Image],
    boxes: torch.Tensor,
    labels: List[str],
    scores: Optional[List[float]] = None,
    box_color: Optional[Tuple[int, int, int]] = None,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    line_width: int = 3,
) -> Image.Image:
    """
    Visualize detection results on image.

    Args:
        image: Input image
        boxes: Boxes in xyxy pixel format (N, 4)
        labels: List of object labels
        scores: Optional list of confidence scores
        box_color: Color for bounding boxes (random if None)
        text_color: Color for text labels
        line_width: Width of bounding box lines

    Returns:
        Image with visualized detections
    """
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image).convert("RGB")
    else:
        image_pil = image.copy()

    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x0, y0, x1, y1 = box.int().tolist()

        # Generate random color if not specified
        if box_color is None:
            color = tuple(np.random.randint(0, 255, size=3).tolist())
        else:
            color = box_color

        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        # Draw label with background
        if scores is not None:
            label_text = f"{label} ({scores[i]:.2f})"
        else:
            label_text = label

        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), label_text, font)
        else:
            w, h = draw.textsize(label_text, font)
            bbox = (x0, y0, w + x0, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label_text, fill=text_color, font=font)

    return image_pil


def detect_objects_from_instruction(
    image: Union[np.ndarray, Image.Image],
    object_names: List[str],
    detector: GroundingDINODetector,
    return_best_only: bool = True,
) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Detect objects mentioned in instruction and return their bounding boxes.

    Args:
        image: Input image
        object_names: List of object names from instruction
        detector: Initialized GroundingDINODetector instance
        return_best_only: If True, return only the best box per object

    Returns:
        Dictionary mapping object names to bounding boxes (xyxy format in pixels)
        If return_best_only=True: {obj_name: tensor(4,) or None}
        If return_best_only=False: {obj_name: tensor(N, 4) or None}
    """
    # Preprocess image to get size
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image

    H, W = image_pil.size[1], image_pil.size[0]  # PIL size is (W, H)

    # Detect objects
    results = detector.detect_batch(image, object_names)

    # Convert to pixel coordinates
    boxes_dict = {}
    for obj_name, obj_results in results.items():
        if return_best_only:
            if obj_results["best_box"] is not None:
                box_normalized = obj_results["best_box"].unsqueeze(0)
                box_pixels = detector.boxes_to_xyxy(box_normalized, (H, W))[0]
                boxes_dict[obj_name] = box_pixels
                print(
                    f"[GroundingDINO] Detected '{obj_name}' at {box_pixels.tolist()} "
                    f"(score: {obj_results['best_score']:.3f})"
                )
            else:
                boxes_dict[obj_name] = None
                print(f"[GroundingDINO] '{obj_name}' not detected")
        else:
            if len(obj_results["boxes"]) > 0:
                boxes_pixels = detector.boxes_to_xyxy(obj_results["boxes"], (H, W))
                boxes_dict[obj_name] = boxes_pixels
                print(
                    f"[GroundingDINO] Detected {len(boxes_pixels)} instances of '{obj_name}'"
                )
            else:
                boxes_dict[obj_name] = None
                print(f"[GroundingDINO] '{obj_name}' not detected")

    return boxes_dict
