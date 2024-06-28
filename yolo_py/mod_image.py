import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pathlib
import os


def generate_color_mapping(num_classes: int) -> List[Tuple[int, ...]]:
    """Generate a unique BGR color for each class

    :param num_classes: The number of classes in the dataset.
    :return:            List of RGB colors for each class.
    """
    cmap = plt.cm.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(i, bytes=True)[:3][::-1] for i in range(num_classes)]
    return [tuple(int(v) for v in c) for c in colors]

def get_recommended_text_size(x1: int, y1: int, x2: int, y2: int) -> float:
    """Get a nice text size for a given bounding box."""
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    diag_length = np.sqrt(bbox_width**2 + bbox_height**2)

    # This follows the heuristic (defined after some visual experiments):
    # - diag_length=100 -> base_font_size=0.4 (min text size)
    # - diag_length=300 -> base_font_size=0.7 (max text size)
    font_size = diag_length * 0.0015 + 0.25
    font_size = max(0.4, font_size)  # Min = 0.4
    font_size = min(0.7, font_size)  # Max = 0.7

    return font_size

def get_recommended_box_thickness(x1: int, y1: int, x2: int, y2: int) -> int:
    """Get a nice box thickness for a given bounding box."""
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    diag_length = np.sqrt(bbox_width**2 + bbox_height**2)

    if diag_length <= 100:
        return 1
    elif diag_length <= 200:
        return 2
    else:
        return 3

def compute_brightness(color: Tuple[int, int, int]) -> float:
    """Computes the brightness of a given color in RGB format. From https://alienryderflex.com/hsp.html

    :param color: A tuple of three integers representing the RGB values of the color.
    :return: The brightness of the color.
    """
    return (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[0]) / 255

def best_text_color(background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Determine the best color for text to be visible on a given background color.

    :param background_color: RGB values of the background color.
    :return: RGB values of the best text color for the given background color.
    """

    # If the brightness is greater than 0.5, use black text; otherwise, use white text.
    if compute_brightness(background_color) > 0.5:
        return (0, 0, 0)  # Black
    else:
        return (255, 255, 255)  # White
    
def draw_text_box(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    font_size: float,
    background_color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    """Draw a text inside a box

    :param image:               The image on which to draw the text box.
    :param text:                The text to display in the text box.
    :param x:                   The x-coordinate of the top-left corner of the text box.
    :param y:                   The y-coordinate of the top-left corner of the text box.
    :param font:                The font to use for the text.
    :param font_size:           The size of the font to use.
    :param background_color:    The color of the text box and text as a tuple of three integers representing RGB values.
    :param thickness:           The thickness of the text.
    :return: Image with the text inside the box.
    """
    text_color = best_text_color(background_color)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, thickness)
    text_left_offset = 7

    image = cv2.rectangle(image, (x, y), (x + text_width + text_left_offset, y - text_height - int(15 * font_size)), background_color, -1)
    image = cv2.putText(image, text, (x + text_left_offset, y - int(10 * font_size)), font, font_size, text_color, thickness, lineType=cv2.LINE_AA)
    return image

def draw_bbox(
    image: np.ndarray,
    title: Optional[str],
    color: Tuple[int, int, int],
    box_thickness: Optional[int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Draw a bounding box on an image.

    :param image:           Image on which to draw the bounding box.
    :param color:           RGB values of the color of the bounding box.
    :param title:           Title to display inside the bounding box.
    :param box_thickness:   Thickness of the bounding box border.
    :param x1:              x-coordinate of the top-left corner of the bounding box.
    :param y1:              y-coordinate of the top-left corner of the bounding box.
    :param x2:              x-coordinate of the bottom-right corner of the bounding box.
    :param y2:              y-coordinate of the bottom-right corner of the bounding box.
    """

    if box_thickness is None:
        box_thickness = get_recommended_box_thickness(x1=x1, y1=y1, x2=x2, y2=y2)

    # Draw bbox
    overlay = image.copy()
    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

    if title is not None or title != "":
        # Adapt font size to image shape.
        # This is required because small images require small font size, but this makes the title look bad,
        # so when possible we increase the font size to a more appropriate value.

        font_size = get_recommended_text_size(x1=x1, y1=y1, x2=x2, y2=y2)
        overlay = draw_text_box(image=overlay, text=title, x=x1, y=y1, font=2, font_size=font_size, background_color=color, thickness=1)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)

def draw_box_title(
        color_mapping: List[Tuple[int]],
        class_names: List[str],
        box_thickness: Optional[int],
        image_np: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        class_id: int,
        pred_conf: float = None,
        bbox_prefix: str = "",
    ):
        """
        Draw a rectangle with class name, confidence on the image
        :param color_mapping: A list of N RGB colors for each class
        :param class_names: A list of N class names
        :param box_thickness: Thickness of the bounding box (in pixels)
        :param image_np: Image in RGB format (H, W, C) where to draw the bounding box
        :param x1: X coordinate of the top left corner of the bounding box
        :param y1: Y coordinate of the top left corner of the bounding box
        :param x2: X coordinate of the bottom right corner of the bounding box
        :param y2: Y coordinate of the bottom right corner of the bounding box
        :param class_id: A corresponding class id
        :param pred_conf: Class confidence score (optional)
        :param bbox_prefix: Prefix to add to the title of the bounding boxes
        """
        color = color_mapping[class_id]
        class_name = class_names[class_id]

        title = class_name
        if bbox_prefix:
            title = f"{bbox_prefix} {class_name}"
        if pred_conf is not None:
            title = f"{title} {str(round(pred_conf, 2))}"

        image_np = draw_bbox(image=image_np, title=title, x1=x1, y1=y1, x2=x2, y2=y2, box_thickness=box_thickness, color=color)
        return image_np

def visualize_image(
        image_np: np.ndarray,
        class_names: List[str],
        target_boxes: Optional[np.ndarray] = None,
        pred_boxes: Optional[np.ndarray] = None,
        box_thickness: Optional[int] = 2,
        gt_alpha: float = 0.6,
        image_scale: float = 1.0,
        checkpoint_dir: Optional[str] = None,
        image_name: Optional[str] = None,
    ):
        image_np = cv2.resize(image_np, (0, 0), fx=image_scale, fy=image_scale, interpolation=cv2.INTER_NEAREST)
        color_mapping = generate_color_mapping(len(class_names))

        if pred_boxes is not None:
            # Draw predictions
            pred_boxes[:, :4] *= image_scale
            for xyxy_score_label in pred_boxes:
                image_np = draw_box_title(
                    color_mapping=color_mapping,
                    class_names=class_names,
                    box_thickness=box_thickness,
                    image_np=image_np,
                    x1=int(xyxy_score_label[0]),
                    y1=int(xyxy_score_label[1]),
                    x2=int(xyxy_score_label[2]),
                    y2=int(xyxy_score_label[3]),
                    class_id=int(xyxy_score_label[5]),
                    pred_conf=float(xyxy_score_label[4]),
                    bbox_prefix="[Pred]" if target_boxes is not None else "",  # If we have TARGETS, we want to add a prefix to distinguish.
                )

        if target_boxes is not None:
            # If gt_alpha is set, we will show it as a transparent overlay.
            if gt_alpha is not None:
                # Transparent overlay of ground truth boxes
                image_with_targets = np.zeros_like(image_np, np.uint8)
            else:
                image_with_targets = image_np

            for label, x1, y1, x2, y2 in target_boxes:
                image_with_targets = draw_box_title(
                    color_mapping=color_mapping,
                    class_names=class_names,
                    box_thickness=box_thickness,
                    image_np=image_with_targets,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=int(label),
                    bbox_prefix="[GT]" if pred_boxes is not None else "",  # If we have PREDICTIONS, we want to add a prefix to distinguish.
                )

            if gt_alpha is not None:
                # Transparent overlay of ground truth boxes
                mask = image_with_targets.astype(bool)
                image_np[mask] = cv2.addWeighted(image_np, 1 - gt_alpha, image_with_targets, gt_alpha, 0)[mask]
            else:
                image_np = image_with_targets

        if checkpoint_dir is None:
            return image_np
        else:
            pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(checkpoint_dir, str(image_name) + ".jpg"), image_np)


