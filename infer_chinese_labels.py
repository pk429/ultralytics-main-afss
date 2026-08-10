# -*- coding: utf-8 -*-
"""YOLO inference with Chinese labels drawn on the top-left of each box.

Examples:
    python infer_chinese_labels.py \
        --model runs/detect/runs/train/FISHER_DETECT_v11s_1280_AFSS_0807/weights/best.pt \
        --source /mnt/sda1/xzm/datasets/fisher/test_datasets/images \
        --labels "fisher:钓鱼,umbrella:遮阳伞"

    python infer_chinese_labels.py \
        --model yolo11s.pt \
        --source input.mp4 \
        --labels "0:钓鱼,1:遮阳伞" \
        --show-conf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO inference and draw Chinese labels.")
    parser.add_argument("--model", required=True, help="YOLO weights path, e.g. runs/.../weights/best.pt")
    parser.add_argument("--source", required=True, help="image/video path or image directory")
    parser.add_argument("--output", default="runs/chinese_predict", help="output directory")
    parser.add_argument("--labels", default="", help='Chinese labels, e.g. "0:钓鱼,1:遮阳伞" or "fisher:钓鱼"')
    parser.add_argument("--font", default=DEFAULT_FONT, help="Chinese font path")
    parser.add_argument("--font-size", type=int, default=26, help="label font size")
    parser.add_argument("--imgsz", type=int, default=1280, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="0", help="device, e.g. 0 or cpu")
    parser.add_argument("--line-width", type=int, default=3, help="box line width")
    parser.add_argument("--show-conf", action="store_true", help="append confidence after label")
    parser.add_argument("--save-txt", action="store_true", help="save YOLO txt labels")
    return parser.parse_args()


def parse_label_map(text: str, model_names: dict | list) -> tuple[dict[int, str], dict[str, str]]:
    """Parse label mapping from 'id:name,class_name:中文' text."""
    by_id: dict[int, str] = {}
    by_name: dict[str, str] = {}
    if not text:
        return by_id, by_name

    for item in text.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        if key.isdigit():
            by_id[int(key)] = value
        else:
            by_name[key] = value

    if by_name:
        names = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
        for cls_id, name in names:
            if str(name) in by_name:
                by_id[int(cls_id)] = by_name[str(name)]
    return by_id, by_name


def load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Load a Chinese-capable font."""
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Font not found: {font_path}. Please pass --font with a Chinese font, "
            f"for example {DEFAULT_FONT}"
        )
    return ImageFont.truetype(str(path), font_size)


def class_name(model_names: dict | list, cls_id: int) -> str:
    """Get class name from Ultralytics model names."""
    if isinstance(model_names, dict):
        return str(model_names.get(cls_id, cls_id))
    if 0 <= cls_id < len(model_names):
        return str(model_names[cls_id])
    return str(cls_id)


def color_for_class(cls_id: int) -> tuple[int, int, int]:
    """Return deterministic RGB color for a class id."""
    palette = [
        (255, 56, 56),
        (56, 128, 255),
        (0, 180, 80),
        (255, 160, 0),
        (180, 70, 255),
        (0, 190, 190),
    ]
    return palette[cls_id % len(palette)]


def draw_chinese_boxes(
    bgr: np.ndarray,
    result,
    model_names: dict | list,
    label_by_id: dict[int, str],
    font: ImageFont.FreeTypeFont,
    line_width: int = 3,
    show_conf: bool = False,
) -> np.ndarray:
    """Draw boxes and Chinese labels on a BGR image."""
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return bgr

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(cls_ids), dtype=np.float32)

    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = color_for_class(cls_id)
        label = label_by_id.get(cls_id, class_name(model_names, cls_id))
        if show_conf:
            label = f"{label} {conf:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        pad_x, pad_y = 6, 4
        label_x1 = max(0, x1)
        label_y1 = max(0, y1 - text_h - 2 * pad_y)
        if label_y1 == 0 and y1 < text_h + 2 * pad_y:
            label_y1 = min(max(0, y1), max(0, image.height - text_h - 2 * pad_y))
        label_x2 = min(image.width, label_x1 + text_w + 2 * pad_x)
        label_y2 = min(image.height, label_y1 + text_h + 2 * pad_y)

        draw.rectangle((label_x1, label_y1, label_x2, label_y2), fill=color)
        draw.text((label_x1 + pad_x, label_y1 + pad_y - top), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def output_path_for_image(source_path: str, output_dir: Path, index: int) -> Path:
    """Build output path for an image result."""
    path = Path(source_path)
    suffix = path.suffix if path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    stem = path.stem if path.stem else f"frame_{index:06d}"
    return output_dir / f"{stem}_cn{suffix}"


def save_yolo_txt(result, txt_path: Path, image_shape: tuple[int, int]) -> None:
    """Save detection boxes as YOLO-format txt."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return
    h, w = image_shape[:2]
    lines = []
    for box, cls_id, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run inference and save annotated outputs."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    font = load_font(args.font, args.font_size)
    label_by_id, _ = parse_label_map(args.labels, model.names)

    source = Path(args.source)
    is_video = source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES
    video_writer = None
    video_out_path = output_dir / f"{source.stem}_cn.mp4" if is_video else None

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        verbose=False,
        save=False,
    )

    for index, result in enumerate(results):
        annotated = draw_chinese_boxes(
            result.orig_img.copy(),
            result,
            model.names,
            label_by_id,
            font,
            line_width=args.line_width,
            show_conf=args.show_conf,
        )

        if is_video:
            if video_writer is None:
                cap = cv2.VideoCapture(str(source))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                cap.release()
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w, h))
            video_writer.write(annotated)
        else:
            image_out = output_path_for_image(result.path, output_dir, index)
            cv2.imwrite(str(image_out), annotated)
            if args.save_txt:
                save_yolo_txt(result, output_dir / "labels" / f"{image_out.stem}.txt", annotated.shape)

    if video_writer is not None:
        video_writer.release()
        print(f"Saved video: {video_out_path}")
    else:
        print(f"Saved images to: {output_dir}")


if __name__ == "__main__":
    main()
