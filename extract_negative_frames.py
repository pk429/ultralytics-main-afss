#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从视频或图片文件夹自动筛选负样本候选：
  - no_det   : 无任何检测框（在 scan_conf 下仍无框，适合做纯负样本）
  - low_conf : 有框但最高置信度 < low_conf（疑似误检，适合做难负样本）
  - high_conf: 有框且最高置信度 >= low_conf（正样本候选，导出模型预测标签供人工复核）

输出：
  manifest.csv / manifest.json  帧清单
  no_det/images, no_det/labels, no_det/json
  low_conf/images, low_conf/labels, low_conf/json
  high_conf/images, high_conf/labels, high_conf/json

示例（视频）：
  python extract_negative_frames.py \
    --video /mnt/sda1/xzm/video/mqtt_wangcheng_video/DJI_20260529160033_0001_V.mp4 \
    --model runs/detect/runs/train/FISHER_DETECT_v11s_p2_1280_0616/weights/best.pt \
    --out runs/neg_mine/fisher_test01 \
    --stride 5 \
    --save-images

示例（图片文件夹）：
  python extract_negative_frames.py \
    --images-dir /mnt/sda1/xzm/datasets/fisher/split/images/val \
    --model runs/detect/runs/train/FISHER_DETECT_v11s_1280_0612/weights/best.pt \
    --out runs/neg_mine/val_scan \
    --save-images
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="从视频或图片文件夹挖掘负样本和高置信正样本候选")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="输入视频路径")
    src.add_argument("--images-dir", help="输入图片文件夹路径")
    p.add_argument("--model", required=True, help="YOLO 权重路径")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--stride", type=int, default=5, help="视频每隔 N 帧采样；文件夹每隔 N 张图采样")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="--images-dir 时递归扫描子目录",
    )
    p.add_argument(
        "--scan-conf",
        type=float,
        default=0.10,
        help="推理置信度下限（尽量低，用于扫出所有可疑框）",
    )
    p.add_argument(
        "--low-conf",
        type=float,
        default=0.35,
        help="低于该最高置信度的帧归为 low_conf（疑似误检）",
    )
    p.add_argument(
        "--classes",
        default="",
        help="只关注这些类别 id，逗号分隔，如 0 表示仅 fisher",
    )
    p.add_argument(
        "--save-images",
        action="store_true",
        help="按类别保存候选帧，并写出对应 labels/*.txt",
    )
    p.add_argument(
        "--categories",
        default="no_det,low_conf,high_conf",
        help="导出哪些类别，逗号分隔：no_det,low_conf,high_conf",
    )
    p.add_argument("--max-save", type=int, default=0, help="每类最多保存张数，0=不限制")
    p.add_argument("--save-width", type=int, default=1920, help="导出图片宽度，0=保持原尺寸")
    p.add_argument("--save-height", type=int, default=1080, help="导出图片高度，0=保持原尺寸")
    p.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="stretch=拉伸到目标分辨率，letterbox=等比缩放后黑边填充",
    )
    p.add_argument("--jpeg-quality", type=int, default=90, help="jpg 压缩质量 1-100")
    return p.parse_args()


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_images(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=natural_sort_key)


def frame_stem(video_stem: str, frame_idx: int) -> str:
    return f"{video_stem}_f{frame_idx:06d}"


def resize_frame(bgr: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    if width <= 0 or height <= 0:
        return bgr
    h0, w0 = bgr.shape[:2]
    if w0 == width and h0 == height:
        return bgr
    if mode == "letterbox":
        scale = min(width / w0, height / h0)
        nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x0 = (width - nw) // 2
        y0 = (height - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        return canvas
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)


def classify_frame(det_count: int, max_conf: float, low_conf: float) -> str:
    if det_count == 0:
        return "no_det"
    if max_conf < low_conf:
        return "low_conf"
    return "high_conf"


def run_predict(model: YOLO, frame: np.ndarray, args) -> tuple[int, float, list[int], list[dict]]:
    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.scan_conf,
        iou=0.5,
        device=args.device,
        verbose=False,
    )[0]
    boxes = results.boxes
    confs: list[float] = []
    cls_ids: list[int] = []
    detections: list[dict] = []
    if boxes is not None and len(boxes):
        for i in range(len(boxes)):
            cid = int(boxes.cls[i])
            if args.class_filter is not None and cid not in args.class_filter:
                continue
            conf = float(boxes.conf[i])
            xyxy = [float(x) for x in boxes.xyxy[i].tolist()]
            confs.append(conf)
            cls_ids.append(cid)
            detections.append(
                {
                    "class_id": cid,
                    "confidence": round(conf, 6),
                    "xyxy": xyxy,
                }
            )
    det_count = len(confs)
    max_conf = max(confs) if confs else 0.0
    return det_count, max_conf, cls_ids, detections


def detection_to_yolo_lines(
    detections: list[dict],
    frame_shape: tuple[int, int, int],
    save_size: tuple[int, int] | None,
    resize_mode: str,
) -> list[str]:
    h0, w0 = frame_shape[:2]
    out_w, out_h = save_size if save_size else (w0, h0)
    lines: list[str] = []

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        if save_size and resize_mode == "letterbox":
            scale = min(out_w / w0, out_h / h0)
            nw, nh = w0 * scale, h0 * scale
            x_pad = (out_w - nw) / 2
            y_pad = (out_h - nh) / 2
            x1, x2 = x1 * scale + x_pad, x2 * scale + x_pad
            y1, y2 = y1 * scale + y_pad, y2 * scale + y_pad
        elif save_size and resize_mode == "stretch":
            x_scale = out_w / w0
            y_scale = out_h / h0
            x1, x2 = x1 * x_scale, x2 * x_scale
            y1, y2 = y1 * y_scale, y2 * y_scale

        x1 = max(0.0, min(out_w, x1))
        x2 = max(0.0, min(out_w, x2))
        y1 = max(0.0, min(out_h, y1))
        y2 = max(0.0, min(out_h, y2))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0 or bh <= 0:
            continue

        cx = (x1 + x2) / 2 / out_w
        cy = (y1 + y2) / 2 / out_h
        nw = bw / out_w
        nh = bh / out_h
        lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def maybe_save_candidate(
    row: dict,
    frame: np.ndarray,
    category: str,
    out_dir: Path,
    save_images: bool,
    save_size: tuple[int, int] | None,
    args,
    saved_count: dict[str, int],
) -> None:
    if category not in args.export_cats:
        return
    if args.max_save > 0 and saved_count[category] >= args.max_save:
        return
    if not save_images:
        return

    stem = row["stem"]
    category_dir = out_dir / category
    img_path = category_dir / "images" / f"{stem}.jpg"
    label_path = category_dir / "labels" / f"{stem}.txt"
    out_frame = frame
    if save_size:
        out_frame = resize_frame(frame, save_size[0], save_size[1], args.resize_mode)
    cv2.imwrite(
        str(img_path),
        out_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, args.jpeg_quality))],
    )
    if category == "no_det":
        label_path.write_text("", encoding="utf-8")
    elif category in {"low_conf", "high_conf"}:
        lines = detection_to_yolo_lines(row["detections"], frame.shape, save_size, args.resize_mode)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    saved_count[category] += 1


def process_video(model: YOLO, video: Path, args, out_dir: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_stem = video.stem
    rows: list[dict] = []
    saved_count: dict[str, int] = {c: 0 for c in args.export_cats}
    save_size = args.save_size

    print(f"模式: 视频", flush=True)
    print(f"视频: {video}", flush=True)
    print(f"总帧数(约): {total}  stride={args.stride}  scan_conf={args.scan_conf}  low_conf={args.low_conf}", flush=True)

    frame_idx = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            det_count, max_conf, cls_ids, detections = run_predict(model, frame, args)
            category = classify_frame(det_count, max_conf, args.low_conf)
            row = {
                "source_type": "video",
                "source": video.name,
                "image": "",
                "frame_idx": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "category": category,
                "det_count": det_count,
                "max_conf": round(max_conf, 4),
                "class_ids": cls_ids,
                "detections": detections,
                "stem": frame_stem(video_stem, frame_idx),
            }
            rows.append(row)
            maybe_save_candidate(row, frame, category, out_dir, args.save_images, save_size, args, saved_count)

            processed += 1
            if processed % 100 == 0:
                print(f"  已处理采样帧 {processed}  (frame_idx={frame_idx}/{total})  saved={saved_count}", flush=True)
            frame_idx += 1
    finally:
        cap.release()

    args.saved_count = saved_count
    return rows


def process_images_dir(model: YOLO, images_dir: Path, args, out_dir: Path) -> list[dict]:
    image_files = list_images(images_dir, args.recursive)
    if not image_files:
        raise FileNotFoundError(f"文件夹内未找到图片: {images_dir}")

    rows: list[dict] = []
    saved_count: dict[str, int] = {c: 0 for c in args.export_cats}
    save_size = args.save_size
    total = len(image_files)

    print(f"模式: 图片文件夹", flush=True)
    print(f"目录: {images_dir}", flush=True)
    print(f"图片数: {total}  stride={args.stride}  recursive={args.recursive}", flush=True)
    print(f"scan_conf={args.scan_conf}  low_conf={args.low_conf}", flush=True)

    processed = 0
    for img_idx, img_path in enumerate(image_files):
        if img_idx % args.stride != 0:
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  跳过无法读取的图片: {img_path}")
            continue

        det_count, max_conf, cls_ids, detections = run_predict(model, frame, args)
        category = classify_frame(det_count, max_conf, args.low_conf)
        row = {
            "source_type": "image",
            "source": str(images_dir),
            "image": img_path.name,
            "frame_idx": img_idx,
            "time_sec": "",
            "category": category,
            "det_count": det_count,
            "max_conf": round(max_conf, 4),
            "class_ids": cls_ids,
            "detections": detections,
            "stem": img_path.stem,
        }
        rows.append(row)
        maybe_save_candidate(row, frame, category, out_dir, args.save_images, save_size, args, saved_count)

        processed += 1
        if processed % 100 == 0:
            print(f"  已处理 {processed} 张  (index={img_idx + 1}/{total})  saved={saved_count}", flush=True)

    args.saved_count = saved_count
    return rows


def write_manifest(rows: list[dict], args, out_dir: Path, input_desc: str) -> None:
    stats = {cat: sum(1 for r in rows if r["category"] == cat) for cat in ("no_det", "low_conf", "high_conf")}

    fieldnames = [
        "source_type",
        "source",
        "image",
        "frame_idx",
        "time_sec",
        "category",
        "det_count",
        "max_conf",
        "class_ids",
        "detections",
        "stem",
    ]
    manifest_csv = out_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if r["category"] not in args.export_cats:
                continue
            r_out = dict(r)
            r_out["class_ids"] = ",".join(map(str, r["class_ids"]))
            r_out["detections"] = json.dumps(r["detections"], ensure_ascii=False)
            writer.writerow(r_out)

    manifest_json = out_dir / "manifest.json"
    summary = {
        "input": input_desc,
        "source_type": "video" if args.video else "image",
        "model": args.model,
        "scan_conf": args.scan_conf,
        "low_conf": args.low_conf,
        "stride": args.stride,
        "sampled_items": len(rows),
        "stats": stats,
        "export_categories": sorted(args.export_cats),
        "saved_images": args.saved_count if args.save_images else {},
        "items": rows,
    }
    manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成。采样数: {len(rows)}", flush=True)
    print(f"  no_det   (无框):     {stats['no_det']}", flush=True)
    print(f"  low_conf (低置信):  {stats['low_conf']}", flush=True)
    print(f"  high_conf(高置信):  {stats['high_conf']}", flush=True)
    print(f"清单: {manifest_csv}", flush=True)
    print(f"详情: {manifest_json}", flush=True)
    if args.save_images:
        for cat in sorted(args.export_cats):
            print(
                f"{cat}: {out_dir / cat / 'images'} | "
                f"{out_dir / cat / 'labels'} | "
                f"{out_dir / cat / 'json'}",
                flush=True,
            )


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.class_filter = None
    if args.classes.strip():
        args.class_filter = {int(x.strip()) for x in args.classes.split(",") if x.strip()}

    args.export_cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    args.save_images = args.save_images
    if args.save_images:
        for cat in args.export_cats:
            (out_dir / cat / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / cat / "labels").mkdir(parents=True, exist_ok=True)
            (out_dir / cat / "json").mkdir(parents=True, exist_ok=True)

    args.save_size = None
    if args.save_width > 0 and args.save_height > 0:
        args.save_size = (args.save_width, args.save_height)

    print(f"导出类别: {sorted(args.export_cats)}  save_images={args.save_images}", flush=True)
    if args.save_images and args.save_size:
        print(f"导出尺寸: {args.save_size[0]}x{args.save_size[1]}  mode={args.resize_mode}  jpeg_q={args.jpeg_quality}", flush=True)

    model = YOLO(args.model)
    args.saved_count = {}

    if args.video:
        video = Path(args.video)
        if not video.is_file():
            raise FileNotFoundError(f"视频不存在: {video}")
        rows = process_video(model, video, args, out_dir)
        write_manifest(rows, args, out_dir, str(video))
    else:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"图片文件夹不存在: {images_dir}")
        rows = process_images_dir(model, images_dir, args, out_dir)
        write_manifest(rows, args, out_dir, str(images_dir))


if __name__ == "__main__":
    main()
