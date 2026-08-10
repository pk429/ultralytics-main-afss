# -*- coding: utf-8 -*-
"""Practical YOLO training workflow.

This script keeps common training modes in one readable entry point:
normal training, incremental fine-tuning, checkpoint resume, validation,
and metric summary.


# 日常训练
python train_yolo_workflow.py --preset floodgate --mode train

# 训练前只做数据体检
python train_yolo_workflow.py --preset floodgate --mode audit

# 抽检标注可视化
python train_yolo_workflow.py --preset floodgate --mode visualize --visualize-samples 100

# 增量训练
python train_yolo_workflow.py --preset swimmer --mode increment \
  --weights /path/to/old_best.pt \
  --data /path/to/mixed_data.yaml \
  --epochs 30

# 新旧模型对比
python train_yolo_workflow.py --preset swimmer --mode compare \
  --old-weights /path/to/old_best.pt \
  --new-weights /path/to/new_best.pt

# 多场景验证
python train_yolo_workflow.py --preset swimmer --mode multi_val \
  --weights /path/to/best.pt \
  --val-set normal=/path/to/normal.yaml \
  --val-set hard=/path/to/hard.yaml

# 难例挖掘
python train_yolo_workflow.py --preset swimmer --mode hard \
  --weights /path/to/best.pt \
  --source /path/to/images

# 导出
python train_yolo_workflow.py --preset swimmer --mode export \
  --weights /path/to/best.pt \
  --export-formats onnx,engine







"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")


# =========================
# 1. Edit These Two Lines
# =========================

ACTIVE_PRESET = "floodgate"
RUN_MODE = "train"


# =========================
# 2. Experiment Presets
# =========================


@dataclass
class TrainPreset:
    """One experiment configuration."""

    data: str
    name: str
    project: str = "runs/train"

    # For normal training. Use a .pt directly, or a model .yaml plus pretrained_weights.
    model_cfg: str = "yolo11s.pt"
    pretrained_weights: str | None = None

    # For incremental training and resume.
    base_checkpoint: str | None = None  # usually old best.pt
    resume_checkpoint: str | None = None  # usually last.pt

    imgsz: int = 1280
    epochs: int = 100
    batch: int = 32
    workers: int = 4
    patience: int = 30
    device: str = "0"
    optimizer: str = "SGD"
    close_mosaic: int = 30
    cache: bool | str = False
    seed: int = 0
    deterministic: bool = True
    cos_lr: bool = False
    amp: bool = True

    # Normal training LR.
    lr0: float | None = None
    lrf: float | None = None

    # Incremental fine-tune LR. Keep it lower than training from scratch.
    increment_epochs: int = 30
    increment_lr0: float = 5e-4
    increment_lrf: float = 0.01
    increment_close_mosaic: int = 10
    increment_freeze: int | list[int] | None = None

    # Practical workflow switches.
    preflight_audit: bool = True
    preflight_visualize: bool = True
    audit_max_images: int = 0  # 0 means all images
    visualize_samples: int = 80
    old_data: str | None = None
    new_data: str | None = None
    mix_new_ratio: float = 0.6
    mix_old_ratio: float = 0.4
    mix_output_dir: str = "runs/dataset_mix"
    val_sets: dict[str, str] = field(default_factory=dict)
    hard_source: str | None = None
    export_formats: tuple[str, ...] = ("onnx",)
    benchmark_export: bool = False

    # Optional common train kwargs. Put rare knobs here instead of scattering code.
    extra_args: dict[str, Any] = field(default_factory=dict)


PRESETS: dict[str, TrainPreset] = {
    "road_pothole_grain": TrainPreset(
        data="./yaml/yolov11_road_pothole_grain.yaml",
        name="ROAD_POTHOLE_GRAIN_DETECT_v11s_1280_0615",
        project="runs/train",
        model_cfg="yolo11s.pt",
        imgsz=1280,
        epochs=200,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
    "road_pothole_grain_resume": TrainPreset(
        data="./yaml/yolov11_road_pothole_grain.yaml",
        name="ROAD_POTHOLE_GRAIN_DETECT_v11s_1280_0625",
        project="runs/train",
        model_cfg="yolo11s.pt",
        resume_checkpoint="/mnt/sda1/xzm/Code/ultralytics-main/runs/detect/runs/train/ROAD_POTHOLE_GRAIN_DETECT_v11s_1280_0625/weights/last.pt",
        imgsz=1280,
        epochs=200,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
    "vehicle_v8n": TrainPreset(
        data="./yaml/yolov8_vechile_detect.yaml",
        name="VECHILE_DETECT_0513_v8n",
        project="runs/train",
        model_cfg="yolov8n.pt",
        imgsz=1920,
        epochs=300,
        batch=32,
        workers=4,
        patience=100,
        close_mosaic=10,
    ),
    "floodgate": TrainPreset(
        data="./yaml/yolov11_flooddgate_detect.yaml",
        name="FLOODGATE_DETECT_v11s_1280_0806",
        project="runs/train",
        model_cfg="yolo11s.pt",
        pretrained_weights=None,
        imgsz=1280,
        epochs=100,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
    "embankment": TrainPreset(
        data="./yaml/yolov11_embankment_detect.yaml",
        name="EMBANKMENT_DETECT_v11s_1280_0622",
        project="runs/train",
        model_cfg="yolo11s.pt",
        imgsz=1280,
        epochs=200,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
    "sand_mining": TrainPreset(
        data="./yaml/yolov11_sand-mining_detect.yaml",
        name="SAND-MINING_DETECT_v11s_1280_0630",
        project="sand-mining-detect",
        model_cfg="yolo11s.pt",
        imgsz=1280,
        epochs=100,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
    "river_rubbish": TrainPreset(
        data="./yaml/yolov11_river_rubbish.yaml",
        name="RIVER_RUBBISH_DETECTED_v11s_1280_0731",
        project="runs/train",
        model_cfg="yolo11s.pt",
        imgsz=1280,
        epochs=100,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
    "swimmer": TrainPreset(
        data="./yaml/yolov11_swimmer_detect.yaml",
        name="SWIMMER_DETECTED_v11s_1280_0805",
        project="runs/train",
        model_cfg="yolo11s.pt",
        pretrained_weights=None,
        base_checkpoint=None,  # set to old best.pt when RUN_MODE="increment"
        imgsz=1280,
        epochs=50,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
        increment_epochs=30,
        increment_lr0=5e-4,
        increment_close_mosaic=10,
    ),
    "cargoship_obb": TrainPreset(
        data="./yaml/yolov11_cargoship_detect.yaml",
        name="CARGOSHIP_DETECT_v11s_1280_0727",
        project="cargoship-obbdetect",
        model_cfg="yolo11s-obb.pt",
        pretrained_weights=None,
        imgsz=1280,
        epochs=200,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
    "cargoship_obb_increment": TrainPreset(
        data="./yaml/yolov11_cargoship_detect.yaml",
        name="CARGOSHIP_DETECT_v11s_1280_0728",
        project="cargoship-obbdetect",
        model_cfg="yolo11s-obb.pt",
        base_checkpoint="/mnt/sda1/xzm/Code/ultralytics-main/runs/obb/cargoship-obbdetect/CARGOSHIP_DETECT_v11s_1280_0727/weights/best.pt",
        imgsz=1280,
        epochs=50,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
        increment_epochs=50,
        increment_lr0=0.001,
        increment_close_mosaic=30,
    ),
    "illegalfish_obb": TrainPreset(
        data="./yaml/yolov11_illegalfish_detect.yaml",
        name="ILLEGALFISH_DETECTED_v11s_1280_0707_ep200",
        project="illegalfish-obbdetect",
        model_cfg="yolo11s-obb.pt",
        imgsz=1280,
        epochs=200,
        batch=32,
        workers=4,
        patience=30,
        close_mosaic=30,
    ),
}


# =========================
# 3. Utility Functions
# =========================


METRIC_PRIORITY = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95(OBB)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50-95(P)",
    "metrics/mAP50(B)",
    "metrics/mAP50(OBB)",
    "metrics/mAP50(M)",
    "metrics/mAP50(P)",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALL_MODES = (
    "train",
    "increment",
    "resume",
    "val",
    "metrics",
    "audit",
    "visualize",
    "mix",
    "compare",
    "multi_val",
    "hard",
    "export",
    "threshold",
)


def as_path(path: str | None) -> Path | None:
    return Path(path).expanduser().resolve() if path else None


def assert_file(path: str | None, label: str) -> Path:
    p = as_path(path)
    if not p or not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return p


def run_dir(cfg: TrainPreset) -> Path:
    return Path(cfg.project).expanduser().resolve() / cfg.name


def latest_last_checkpoint(cfg: TrainPreset) -> Path | None:
    direct = run_dir(cfg) / "weights" / "last.pt"
    if direct.exists():
        return direct
    root = Path(cfg.project).expanduser().resolve()
    candidates = sorted(root.glob("**/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def workflow_dir(cfg: TrainPreset, kind: str) -> Path:
    path = Path("runs") / kind / cfg.name
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    p = assert_file(str(path), "yaml")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境缺少 PyYAML，无法读取 data yaml") from exc
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"yaml 内容不是字典: {p}")
    data["_yaml_file"] = str(p)
    return data


def dump_yaml_file(data: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境缺少 PyYAML，无法写入 data yaml") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, allow_unicode=True, sort_keys=False)


def dataset_base(data_yaml: dict[str, Any]) -> Path:
    yaml_file = Path(data_yaml["_yaml_file"]).resolve()
    root = data_yaml.get("path")
    if root:
        root_path = Path(str(root)).expanduser()
        return (yaml_file.parent / root_path).resolve() if not root_path.is_absolute() else root_path.resolve()
    return yaml_file.parent.resolve()


def class_names(data_yaml: dict[str, Any]) -> dict[int, str]:
    names = data_yaml.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def resolve_entry_paths(entry: Any, base: Path) -> list[Path]:
    if entry is None:
        return []
    if isinstance(entry, (list, tuple)):
        paths: list[Path] = []
        for item in entry:
            paths.extend(resolve_entry_paths(item, base))
        return paths

    raw = str(entry).strip()
    if not raw:
        return []
    p = Path(raw).expanduser()
    p = p if p.is_absolute() else base / p

    if p.is_file() and p.suffix.lower() == ".txt":
        paths = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                q = Path(line).expanduser()
                paths.append(q if q.is_absolute() else base / q)
        return [x.resolve() for x in paths]

    if any(ch in str(p) for ch in "*?[]"):
        return sorted(x.resolve() for x in p.parent.glob(p.name) if x.suffix.lower() in IMAGE_SUFFIXES)

    if p.is_dir():
        return sorted(x.resolve() for x in p.rglob("*") if x.suffix.lower() in IMAGE_SUFFIXES)

    return [p.resolve()] if p.suffix.lower() in IMAGE_SUFFIXES else []


def image_paths_from_data(data_yaml_path: str | Path, split: str) -> list[Path]:
    data = load_yaml_file(data_yaml_path)
    return resolve_entry_paths(data.get(split), dataset_base(data))


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def parse_label_line(line: str, nc: int | None) -> tuple[int | None, float | None, str | None]:
    parts = line.split()
    if len(parts) < 5:
        return None, None, "字段数少于 5"
    try:
        cls = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None, None, "类别或坐标不是数字"
    if nc is not None and not 0 <= cls < nc:
        return cls, None, f"类别 id 越界: {cls}"
    eps = 1e-3
    if any(v < -eps or v > 1 + eps for v in coords):
        return cls, None, "坐标超出 0~1"
    if len(coords) >= 8 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        area = max(xs) - min(xs)
        area *= max(ys) - min(ys)
        return cls, area, None
    x, y, w, h = coords[:4]
    if w <= 0 or h <= 0:
        return cls, None, "框宽高小于等于 0"
    if x - w / 2 < -eps or x + w / 2 > 1 + eps or y - h / 2 < -eps or y + h / 2 > 1 + eps:
        return cls, w * h, "框边界超出图像"
    return cls, w * h, None


def audit_dataset(cfg: TrainPreset) -> Path:
    data = load_yaml_file(cfg.data)
    names = class_names(data)
    nc = int(data.get("nc", len(names))) if data.get("nc", len(names)) is not None else None
    save_dir = workflow_dir(cfg, "audit")
    report_path = save_dir / "dataset_report.txt"
    class_counts = {i: 0 for i in range(nc or 0)}
    area_bins = {"small": 0, "medium": 0, "large": 0}
    rows: list[str] = []
    problems: list[str] = []
    seen_images: set[Path] = set()
    duplicate_images = 0
    unreadable_images = 0
    try:
        from PIL import Image
    except ModuleNotFoundError:
        Image = None

    for split in ("train", "val", "test"):
        images = image_paths_from_data(cfg.data, split)
        if cfg.audit_max_images > 0:
            images = images[: cfg.audit_max_images]
        total_labels = missing = empty = invalid = 0
        for img in images:
            if img in seen_images:
                duplicate_images += 1
                problems.append(f"[{split}] 重复图片: {img}")
            seen_images.add(img)
            if not img.exists():
                unreadable_images += 1
                problems.append(f"[{split}] 图片不存在: {img}")
                continue
            if Image is not None:
                try:
                    with Image.open(img) as im:
                        im.verify()
                except OSError:
                    unreadable_images += 1
                    problems.append(f"[{split}] 图片无法读取: {img}")
                    continue
            label = label_path_for_image(img)
            if not label.exists():
                missing += 1
                problems.append(f"[{split}] 缺少标签: {label}")
                continue
            lines = [x.strip() for x in label.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            if not lines:
                empty += 1
            for line in lines:
                cls, area, err = parse_label_line(line, nc)
                if cls is not None and cls in class_counts:
                    class_counts[cls] += 1
                if area is not None:
                    total_labels += 1
                    if area < 0.01:
                        area_bins["small"] += 1
                    elif area < 0.05:
                        area_bins["medium"] += 1
                    else:
                        area_bins["large"] += 1
                if err:
                    invalid += 1
                    problems.append(f"[{split}] {label}: {err} -> {line}")
        rows.append(
            f"{split}: images={len(images)}, labels={total_labels}, missing_txt={missing}, empty_txt={empty}, invalid={invalid}"
        )

    lines = [
        f"data: {Path(cfg.data).resolve()}",
        f"names: {names}",
        "",
        "split summary:",
        *rows,
        "",
        "class counts:",
    ]
    for cls, count in sorted(class_counts.items()):
        lines.append(f"{cls} {names.get(cls, str(cls))}: {count}")
    lines.extend(
        [
            "",
            f"bbox area bins(normalized): {area_bins}",
            f"duplicate images: {duplicate_images}",
            f"unreadable images: {unreadable_images}",
            "",
            f"problem count: {len(problems)}",
            *problems[:500],
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (save_dir / "class_counts.csv").write_text(
        "class_id,class_name,count\n"
        + "\n".join(f"{cls},{names.get(cls, str(cls))},{count}" for cls, count in sorted(class_counts.items())),
        encoding="utf-8",
    )
    print(f"数据体检报告: {report_path}")
    return report_path


def draw_label_preview(cfg: TrainPreset) -> Path:
    data = load_yaml_file(cfg.data)
    names = class_names(data)
    images = image_paths_from_data(cfg.data, "train") + image_paths_from_data(cfg.data, "val")
    if not images:
        raise FileNotFoundError("没有找到可视化抽检图片")
    random.seed(cfg.seed)
    sample = random.sample(images, min(cfg.visualize_samples, len(images)))
    save_dir = workflow_dir(cfg, "check_labels")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境缺少 Pillow，无法绘制标注抽检图") from exc

    for img_path in sample:
        label = label_path_for_image(img_path)
        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            continue
        draw = ImageDraw.Draw(image)
        w, h = image.size
        for line in label.read_text(encoding="utf-8", errors="ignore").splitlines() if label.exists() else []:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            color = (255, 60 + cls * 37 % 160, 30 + cls * 71 % 180)
            if len(coords) >= 8 and len(coords) % 2 == 0:
                pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
                draw.line(pts + [pts[0]], fill=color, width=3)
                x1, y1 = pts[0]
            else:
                x, y, bw, bh = coords[:4]
                x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
                x2, y2 = (x + bw / 2) * w, (y + bh / 2) * h
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1, max(0, y1 - 14)), names.get(cls, str(cls)), fill=color)
        out = save_dir / img_path.name
        image.save(out, quality=92)
    print(f"标注抽检图已保存: {save_dir}")
    return save_dir


def preflight(cfg: TrainPreset) -> None:
    if cfg.preflight_audit:
        audit_dataset(cfg)
    if cfg.preflight_visualize:
        draw_label_preview(cfg)


def train_args(cfg: TrainPreset, *, incremental: bool = False) -> dict[str, Any]:
    args: dict[str, Any] = {
        "data": cfg.data,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "workers": cfg.workers,
        "patience": cfg.patience,
        "device": cfg.device,
        "optimizer": cfg.optimizer,
        "project": cfg.project,
        "name": cfg.name,
        "cache": cfg.cache,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "cos_lr": cfg.cos_lr,
        "amp": cfg.amp,
        "resume": False,
        "plots": True,
    }

    if incremental:
        args.update(
            {
                "epochs": cfg.increment_epochs,
                "lr0": cfg.increment_lr0,
                "lrf": cfg.increment_lrf,
                "close_mosaic": cfg.increment_close_mosaic,
            }
        )
        if cfg.increment_freeze is not None:
            args["freeze"] = cfg.increment_freeze
    else:
        args.update({"epochs": cfg.epochs, "close_mosaic": cfg.close_mosaic})
        if cfg.lr0 is not None:
            args["lr0"] = cfg.lr0
        if cfg.lrf is not None:
            args["lrf"] = cfg.lrf

    args.update(cfg.extra_args)
    return args


def print_env(cfg: TrainPreset, mode: str) -> None:
    print("=" * 80)
    print(f"模式: {mode}")
    print(f"实验: {cfg.name}")
    print(f"数据: {cfg.data}")
    print(f"输出: {run_dir(cfg)}")
    try:
        import torch

        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ModuleNotFoundError:
        print("CUDA: 当前 Python 环境未安装 torch，训练前请先激活正确环境")
    print("=" * 80)


def load_yolo_class():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前 Python 环境缺少 ultralytics/torch，请先激活训练环境，例如 conda activate xzm") from exc
    return YOLO


def snapshot_script(save_dir: Path) -> None:
    """Copy this script into the run directory for reproducibility."""
    save_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    dst = save_dir / f"{src.stem}_snapshot.py"
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        print(f"保存脚本快照失败，不影响训练: {exc}")


def build_model_for_train(cfg: TrainPreset) -> YOLO:
    assert_file(cfg.data, "data yaml")
    model_cfg = assert_file(cfg.model_cfg, "model cfg/weights")
    YOLO = load_yolo_class()
    model = YOLO(str(model_cfg))
    if cfg.pretrained_weights:
        pretrained = assert_file(cfg.pretrained_weights, "pretrained weights")
        model.load(str(pretrained))
    return model


def build_model_for_increment(cfg: TrainPreset) -> YOLO:
    assert_file(cfg.data, "data yaml")
    ckpt = assert_file(cfg.base_checkpoint, "base checkpoint(best.pt)")
    YOLO = load_yolo_class()
    return YOLO(str(ckpt))


def train_normal(cfg: TrainPreset) -> None:
    preflight(cfg)
    model = build_model_for_train(cfg)
    model.train(**train_args(cfg, incremental=False))
    save_dir = Path(model.trainer.save_dir).resolve()
    snapshot_script(save_dir)
    summarize_run(save_dir)


def train_incremental(cfg: TrainPreset) -> None:
    preflight(cfg)
    model = build_model_for_increment(cfg)
    old_weights = cfg.base_checkpoint
    model.train(**train_args(cfg, incremental=True))
    save_dir = Path(model.trainer.save_dir).resolve()
    snapshot_script(save_dir)
    summarize_run(save_dir)
    if old_weights:
        compare_models(cfg, old_weights=old_weights, new_weights=str(save_dir / "weights" / "best.pt"))


def train_resume(cfg: TrainPreset) -> None:
    ckpt = as_path(cfg.resume_checkpoint) if cfg.resume_checkpoint else latest_last_checkpoint(cfg)
    if not ckpt or not ckpt.exists():
        raise FileNotFoundError("没有找到可续训的 last.pt，请设置 resume_checkpoint")
    YOLO = load_yolo_class()
    model = YOLO(str(ckpt))
    model.train(resume=True, device=cfg.device, workers=cfg.workers, batch=cfg.batch)
    summarize_run(ckpt.parents[1])


def validate(cfg: TrainPreset) -> None:
    ckpt = as_path(cfg.base_checkpoint) or run_dir(cfg) / "weights" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"验证权重不存在: {ckpt}")
    assert_file(cfg.data, "data yaml")
    YOLO = load_yolo_class()
    model = YOLO(str(ckpt))
    model.val(data=cfg.data, imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device, plots=True)


def metrics_to_dict(metrics: Any) -> dict[str, float]:
    """Extract common metrics from Ultralytics result objects across tasks."""
    out: dict[str, float] = {}
    target = getattr(metrics, "box", None) or getattr(metrics, "obb", None) or getattr(metrics, "seg", None)
    if target is not None:
        for key in ("mp", "mr", "map50", "map", "map75"):
            value = getattr(target, key, None)
            if value is not None:
                out[key] = float(value)
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                out[key] = float(value)
    return out


def val_model(model_path: Path, data_yaml: str, cfg: TrainPreset, *, conf: float | None = None) -> dict[str, float]:
    YOLO = load_yolo_class()
    model = YOLO(str(model_path))
    kwargs = {
        "data": data_yaml,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "device": cfg.device,
        "plots": False,
    }
    if conf is not None:
        kwargs["conf"] = conf
    metrics = model.val(**kwargs)
    return metrics_to_dict(metrics)


def metric_score(metrics: dict[str, float]) -> float:
    for key in ("map", "metrics/mAP50-95(B)", "metrics/mAP50-95(OBB)", "map50", "metrics/mAP50(B)"):
        if key in metrics:
            return metrics[key]
    return 0.0


def compare_models(cfg: TrainPreset, old_weights: str | None = None, new_weights: str | None = None) -> Path:
    old_path = assert_file(old_weights or cfg.base_checkpoint, "old model")
    new_path = assert_file(new_weights or str(run_dir(cfg) / "weights" / "best.pt"), "new model")
    assert_file(cfg.data, "data yaml")
    old_metrics = val_model(old_path, cfg.data, cfg)
    new_metrics = val_model(new_path, cfg.data, cfg)
    keys = sorted(set(old_metrics) | set(new_metrics))
    rows = ["metric,old,new,delta"]
    for key in keys:
        old = old_metrics.get(key)
        new = new_metrics.get(key)
        if old is None or new is None:
            continue
        rows.append(f"{key},{old:.6f},{new:.6f},{new - old:.6f}")
    save_dir = workflow_dir(cfg, "compare")
    out = save_dir / "old_vs_new_metrics.csv"
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"旧新模型对比: {out}")
    return out


def multi_validate(cfg: TrainPreset) -> Path:
    weights = as_path(cfg.base_checkpoint) or run_dir(cfg) / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"验证权重不存在: {weights}")
    val_sets = cfg.val_sets or {"default": cfg.data}
    rows = ["val_set,metric,value"]
    summary: dict[str, dict[str, float]] = {}
    for name, data_yaml in val_sets.items():
        print(f"验证场景: {name} -> {data_yaml}")
        metrics = val_model(weights, data_yaml, cfg)
        summary[name] = metrics
        for key, value in sorted(metrics.items()):
            rows.append(f"{name},{key},{value:.6f}")
    save_dir = workflow_dir(cfg, "multi_val")
    out_csv = save_dir / "multi_val_metrics.csv"
    out_json = save_dir / "multi_val_metrics.json"
    out_csv.write_text("\n".join(rows), encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"多验证集结果: {out_csv}")
    return out_csv


def make_increment_mix(cfg: TrainPreset) -> Path:
    old_data = assert_file(cfg.old_data, "old data yaml")
    new_data = assert_file(cfg.new_data, "new data yaml")
    old_yaml = load_yaml_file(old_data)
    new_yaml = load_yaml_file(new_data)
    old_images = image_paths_from_data(old_data, "train")
    new_images = image_paths_from_data(new_data, "train")
    if not old_images or not new_images:
        raise FileNotFoundError("old/new data 的 train 图片不能为空")

    random.seed(cfg.seed)
    new_count = len(new_images)
    old_count = min(len(old_images), max(1, int(new_count * cfg.mix_old_ratio / max(cfg.mix_new_ratio, 1e-6))))
    mixed = list(new_images) + random.sample(old_images, old_count)
    random.shuffle(mixed)

    out_dir = Path(cfg.mix_output_dir).expanduser().resolve() / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train_mix.txt"
    train_txt.write_text("\n".join(str(p) for p in mixed), encoding="utf-8")

    mixed_yaml = dict(new_yaml)
    mixed_yaml.pop("_yaml_file", None)
    mixed_yaml["path"] = str(out_dir)
    mixed_yaml["train"] = str(train_txt)
    if "val" not in mixed_yaml and old_yaml.get("val"):
        mixed_yaml["val"] = old_yaml["val"]
    out_yaml = out_dir / "data_increment_mix.yaml"
    dump_yaml_file(mixed_yaml, out_yaml)

    report = {
        "new_train_images": len(new_images),
        "old_replay_images": old_count,
        "mixed_train_images": len(mixed),
        "new_ratio_config": cfg.mix_new_ratio,
        "old_ratio_config": cfg.mix_old_ratio,
    }
    (out_dir / "mix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"增量混合 data yaml: {out_yaml}")
    return out_yaml


def iou_xywhn(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def read_gt_boxes(image_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    label = label_path_for_image(image_path)
    if not label.exists():
        return []
    boxes = []
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(coords) >= 4:
            boxes.append((cls, tuple(coords[:4])))
    return boxes


def mine_hard_cases(cfg: TrainPreset) -> Path:
    weights = as_path(cfg.base_checkpoint) or run_dir(cfg) / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"难例挖掘权重不存在: {weights}")
    if cfg.hard_source:
        source_images = resolve_entry_paths(cfg.hard_source, Path.cwd())
    else:
        source_images = image_paths_from_data(cfg.data, "val")
    if not source_images:
        raise FileNotFoundError("难例挖掘没有找到图片源")
    save_dir = workflow_dir(cfg, "hard_cases")
    YOLO = load_yolo_class()
    model = YOLO(str(weights))
    results = model.predict(
        source=[str(p) for p in source_images],
        imgsz=cfg.imgsz,
        device=cfg.device,
        conf=0.15,
        iou=0.6,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(save_dir),
        name="predictions",
        exist_ok=True,
        stream=True,
    )
    hard_rows = ["image,type,class_id,conf,iou,x,y,w,h"]
    match_iou = 0.5
    high_conf = 0.5
    for result in results:
        image_path = Path(result.path).resolve()
        gt_boxes = read_gt_boxes(image_path)
        matched_gt: set[int] = set()
        pred_boxes = getattr(result, "boxes", None)
        if pred_boxes is not None and len(pred_boxes):
            xywhn = pred_boxes.xywhn.cpu().tolist()
            cls_list = [int(x) for x in pred_boxes.cls.cpu().tolist()]
            conf_list = [float(x) for x in pred_boxes.conf.cpu().tolist()]
            for pred_box, pred_cls, pred_conf in zip(xywhn, cls_list, conf_list):
                best_iou, best_idx = 0.0, -1
                pred_tuple = tuple(float(v) for v in pred_box[:4])
                for idx, (gt_cls, gt_box) in enumerate(gt_boxes):
                    if gt_cls != pred_cls:
                        continue
                    overlap = iou_xywhn(pred_tuple, gt_box)
                    if overlap > best_iou:
                        best_iou, best_idx = overlap, idx
                if best_iou >= match_iou:
                    matched_gt.add(best_idx)
                elif pred_conf >= high_conf:
                    hard_rows.append(
                        f"{image_path},false_positive,{pred_cls},{pred_conf:.4f},{best_iou:.4f},"
                        + ",".join(f"{v:.6f}" for v in pred_tuple)
                    )
        for idx, (gt_cls, gt_box) in enumerate(gt_boxes):
            if idx not in matched_gt:
                hard_rows.append(
                    f"{image_path},false_negative,{gt_cls},0.0000,0.0000,"
                    + ",".join(f"{v:.6f}" for v in gt_box)
                )
    hard_csv = save_dir / "hard_candidates.csv"
    hard_csv.write_text("\n".join(hard_rows), encoding="utf-8")
    print(f"难例候选预测结果: {save_dir / 'predictions'}")
    print(f"难例候选 CSV: {hard_csv}")
    return save_dir / "predictions"


def export_and_benchmark(cfg: TrainPreset) -> Path:
    weights = as_path(cfg.base_checkpoint) or run_dir(cfg) / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"导出权重不存在: {weights}")
    YOLO = load_yolo_class()
    model = YOLO(str(weights))
    save_dir = workflow_dir(cfg, "export")
    rows = ["format,output"]
    for fmt in cfg.export_formats:
        print(f"导出格式: {fmt}")
        exported = model.export(format=fmt, imgsz=cfg.imgsz, device=cfg.device, simplify=True)
        rows.append(f"{fmt},{exported}")
    if cfg.benchmark_export:
        result = model.benchmark(data=cfg.data, imgsz=cfg.imgsz, device=cfg.device, verbose=False)
        (save_dir / "benchmark.json").write_text(json.dumps(str(result), ensure_ascii=False, indent=2), encoding="utf-8")
    out = save_dir / "export_report.csv"
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"导出报告: {out}")
    return out


def search_threshold(cfg: TrainPreset) -> Path:
    weights = as_path(cfg.base_checkpoint) or run_dir(cfg) / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"阈值搜索权重不存在: {weights}")
    confs = [round(x / 100, 2) for x in range(5, 91, 5)]
    rows = ["conf,score,map50,map50_95,precision,recall"]
    best = (-1.0, None, {})
    for conf in confs:
        metrics = val_model(weights, cfg.data, cfg, conf=conf)
        score = metric_score(metrics)
        if score > best[0]:
            best = (score, conf, metrics)
        rows.append(
            f"{conf:.2f},{score:.6f},{metrics.get('map50', 0):.6f},{metrics.get('map', 0):.6f},"
            f"{metrics.get('mp', 0):.6f},{metrics.get('mr', 0):.6f}"
        )
    save_dir = workflow_dir(cfg, "threshold")
    out = save_dir / "threshold_search.csv"
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"阈值搜索结果: {out}")
    print(f"推荐 conf: {best[1]}, score={best[0]:.6f}")
    return out


def read_results_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def choose_metric(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    keys = rows[0].keys()
    for key in METRIC_PRIORITY:
        if key in keys:
            return key
    return None


def summarize_run(run_path: Path) -> None:
    csv_path = run_path / "results.csv"
    rows = read_results_csv(csv_path)
    if not rows:
        print(f"没有找到训练指标文件: {csv_path}")
        return

    metric_key = choose_metric(rows)
    summary: dict[str, Any] = {
        "run_path": str(run_path),
        "results_csv": str(csv_path),
        "epochs_recorded": len(rows),
        "main_metric": metric_key,
    }
    print("\n训练指标摘要")
    print("-" * 80)
    print(f"结果目录: {run_path}")
    print(f"指标文件: {csv_path}")
    print(f"总 epoch 记录数: {len(rows)}")

    if metric_key:
        scored = [(idx, to_float(row.get(metric_key))) for idx, row in enumerate(rows, start=1)]
        scored = [(idx, score) for idx, score in scored if score is not None]
        if scored:
            best_epoch, best_score = max(scored, key=lambda x: x[1])
            last_score = scored[-1][1]
            print(f"主指标: {metric_key}")
            print(f"最佳 epoch: {best_epoch}, best={best_score:.5f}")
            print(f"最后 epoch: {len(rows)}, last={last_score:.5f}")
            summary.update(
                {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "last_epoch": len(rows),
                    "last_score": last_score,
                    "best_last_delta": last_score - best_score,
                }
            )
            if len(scored) >= 5:
                recent = [score for _, score in scored[-5:]]
                print("最近 5 个 epoch:", ", ".join(f"{v:.5f}" for v in recent))
                summary["recent_5"] = recent
    else:
        print("没有识别到 mAP 指标列，请手动查看 results.csv")

    important = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    )
    last = rows[-1]
    for key in important:
        value = to_float(last.get(key))
        if value is not None:
            print(f"{key}: {value:.5f}")
            summary[key] = value

    print("\n常用观察文件:")
    for name in ("results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"):
        p = run_path / name
        if p.exists():
            print(f"- {p}")
    summary_path = run_path / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"摘要 JSON: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 日常训练/增量训练/断点续训脚本")
    parser.add_argument("--preset", default=ACTIVE_PRESET, choices=sorted(PRESETS), help="实验配置名")
    parser.add_argument("--mode", default=RUN_MODE, choices=ALL_MODES)
    parser.add_argument("--data", default=None, help="临时覆盖 data yaml")
    parser.add_argument("--model", default=None, help="临时覆盖 model yaml")
    parser.add_argument("--weights", default=None, help="临时覆盖 pretrained/base/resume 权重")
    parser.add_argument("--old-weights", default=None, help="compare 模式旧模型权重")
    parser.add_argument("--new-weights", default=None, help="compare 模式新模型权重")
    parser.add_argument("--old-data", default=None, help="mix 模式旧数据 data yaml")
    parser.add_argument("--new-data", default=None, help="mix 模式新数据 data yaml")
    parser.add_argument("--source", default=None, help="hard 模式预测数据源")
    parser.add_argument("--val-set", action="append", default=None, help="多验证集，格式 name=/path/data.yaml，可重复传入")
    parser.add_argument("--export-formats", default=None, help="导出格式，逗号分隔，如 onnx,engine")
    parser.add_argument("--name", default=None, help="临时覆盖实验名")
    parser.add_argument("--epochs", type=int, default=None, help="临时覆盖训练 epoch")
    parser.add_argument("--imgsz", type=int, default=None, help="临时覆盖输入尺寸")
    parser.add_argument("--batch", type=int, default=None, help="临时覆盖 batch")
    parser.add_argument("--device", default=None, help="临时覆盖 device")
    parser.add_argument("--no-preflight", action="store_true", help="训练前跳过数据体检和标注抽检")
    parser.add_argument("--no-visualize", action="store_true", help="训练前只体检，不生成标注抽检图")
    parser.add_argument("--visualize-samples", type=int, default=None, help="标注抽检图片数量")
    parser.add_argument("--audit-max-images", type=int, default=None, help="体检最多检查多少张，0 为全部")
    parser.add_argument("--benchmark", action="store_true", help="export 模式同时跑 benchmark")
    return parser.parse_args()


def apply_cli_overrides(cfg: TrainPreset, args: argparse.Namespace) -> TrainPreset:
    if args.data:
        cfg.data = args.data
    if args.model:
        cfg.model_cfg = args.model
    if args.weights:
        if args.mode == "train":
            cfg.pretrained_weights = args.weights
        elif args.mode == "increment":
            cfg.base_checkpoint = args.weights
        elif args.mode in {"resume", "val", "multi_val", "hard", "export", "threshold"}:
            cfg.resume_checkpoint = args.weights
            cfg.base_checkpoint = args.weights
    if args.old_weights:
        cfg.base_checkpoint = args.old_weights
    if args.old_data:
        cfg.old_data = args.old_data
    if args.new_data:
        cfg.new_data = args.new_data
    if args.source:
        cfg.hard_source = args.source
    if args.val_set:
        val_sets = {}
        for item in args.val_set:
            if "=" not in item:
                raise ValueError("--val-set 格式必须是 name=/path/data.yaml")
            key, value = item.split("=", 1)
            val_sets[key.strip()] = value.strip()
        cfg.val_sets = val_sets
    if args.export_formats:
        cfg.export_formats = tuple(x.strip() for x in args.export_formats.split(",") if x.strip())
    if args.name:
        cfg.name = args.name
    if args.epochs is not None:
        if args.mode == "increment":
            cfg.increment_epochs = args.epochs
        else:
            cfg.epochs = args.epochs
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.batch is not None:
        cfg.batch = args.batch
    if args.device is not None:
        cfg.device = args.device
    if args.no_preflight:
        cfg.preflight_audit = False
        cfg.preflight_visualize = False
    if args.no_visualize:
        cfg.preflight_visualize = False
    if args.visualize_samples is not None:
        cfg.visualize_samples = args.visualize_samples
    if args.audit_max_images is not None:
        cfg.audit_max_images = args.audit_max_images
    if args.benchmark:
        cfg.benchmark_export = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(PRESETS[args.preset], args)
    print_env(cfg, args.mode)

    if args.mode == "train":
        train_normal(cfg)
    elif args.mode == "increment":
        train_incremental(cfg)
    elif args.mode == "resume":
        train_resume(cfg)
    elif args.mode == "val":
        validate(cfg)
    elif args.mode == "metrics":
        summarize_run(run_dir(cfg))
    elif args.mode == "audit":
        audit_dataset(cfg)
    elif args.mode == "visualize":
        draw_label_preview(cfg)
    elif args.mode == "mix":
        make_increment_mix(cfg)
    elif args.mode == "compare":
        compare_models(cfg, old_weights=args.old_weights or cfg.base_checkpoint, new_weights=args.new_weights)
    elif args.mode == "multi_val":
        multi_validate(cfg)
    elif args.mode == "hard":
        mine_hard_cases(cfg)
    elif args.mode == "export":
        export_and_benchmark(cfg)
    elif args.mode == "threshold":
        search_threshold(cfg)
    else:
        raise ValueError(f"未知模式: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n运行失败: {exc}", file=sys.stderr)
        raise
