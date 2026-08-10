"""YOLO 验证脚本：支持 yaml 或 images/labels 文件夹。"""

# """ascend版本"""
# from ultralytics.nn.modules import Conv
# import torch.nn as nn

# Conv.default_act = nn.Hardswish()

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

# ── 默认配置（也可用命令行覆盖）──
DATA_ROOT = ""  # 填数据根目录则用 images/labels；留空则用 DATA_YAML
DATA_YAML = "//mnt/sda1/xzm/Code/ultralytics-main/yaml/yolov11_cargoship_detect.yaml"
VAL_SUBDIR = "val"  # 例如 val → images/val；留空 → 直接用 images/

CLASS_NAMES =['cargoship']

models = [
    ("v11s_1280_afss", "/mnt/sda1/xzm/Code/ultralytics-main-afss/runs/detect/runs/train/FISHER_DETECT_v11s_1280_AFSS_0807/weights/best.pt", 1280),
    ("v11s_1280", "/mnt/sda1/xzm/Code/ultralytics-main/runs/obb/cargoship-obbdetect/CARGOSHIP_DETECT_v11s_1280_0724_ascend/weights/best.pt", 1280)
]


def infer_data_root_from_yaml(data_yaml: str) -> Path | None:
    """从 yaml 的 train/val/test 路径推断 split 根目录（含 images/、labels/）。"""
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    for key in ("val", "train", "test"):
        raw = cfg.get(key)
        if not raw:
            continue
        p = Path(raw)
        if p.name in {"train", "val", "test"} and p.parent.name == "images":
            return p.parent.parent
        if p.parent.name == "images":
            return p.parent.parent
    return None


def build_data_yaml(data_root: str, names: list[str], subdir: str = "") -> str:
    """根据 images/ + labels/ 目录生成临时 yaml（YOLO 会自动把 images 路径映射到 labels）。"""
    root = Path(data_root)
    images = root / "images" / subdir if subdir else root / "images"
    labels = root / "labels" / subdir if subdir else root / "labels"
    if not images.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {images}")
    if not labels.is_dir():
        raise FileNotFoundError(f"标签目录不存在: {labels}")

    img_path = str(images.resolve())
    cfg = {
        "path": str(root.resolve()),
        "train": img_path,  # Ultralytics yaml 要求必须有 train；仅 val 时与 val 指向同一路径即可
        "val": img_path,
        "nc": len(names),
        "names": names,
    }
    out = Path("runs/val_data_auto.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已生成数据配置: {out}")
    print(f"  val images: {images}")
    print(f"  val labels: {labels}")
    return str(out)


def resolve_data_yaml(data_root: str | None, data_yaml: str | None, names: list[str], subdir: str) -> str:
    if data_root:
        return build_data_yaml(data_root, names, subdir)
    if data_yaml and Path(data_yaml).is_file():
        return data_yaml
    raise ValueError("请指定 --data-root（含 images/labels）或 --data-yaml")


def parse_args():
    p = argparse.ArgumentParser(description="YOLO val：支持 yaml 或 images/labels 文件夹")
    p.add_argument("--data-root", default=None, help="数据根目录，下有 images/ 与 labels/")
    p.add_argument("--data-yaml", default=None, help="已有 yaml（与 --data-root 二选一，优先 data-root）")
    p.add_argument("--subdir", default=None, help="子目录名，如 val → images/val + labels/val")
    p.add_argument("--names", default=None, help="类别名，逗号分隔，如 fisher,umbrella")
    p.add_argument("--model", default=None, help="单个权重路径，指定后忽略下方 models 列表")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_root = args.data_root or DATA_ROOT or None
    data_yaml = args.data_yaml or DATA_YAML or None
    subdir = args.subdir if args.subdir is not None else VAL_SUBDIR
    names = [s.strip() for s in args.names.split(",")] if args.names else CLASS_NAMES

    if data_root:
        data = resolve_data_yaml(data_root, None, names, subdir)
    elif data_yaml:
        if subdir:
            inferred_root = infer_data_root_from_yaml(data_yaml)
            if inferred_root and inferred_root.is_dir():
                print(f"已从 yaml 推断 data_root: {inferred_root}")
                print(f"使用 subdir={subdir!r} → images/{subdir} + labels/{subdir}")
                data = build_data_yaml(str(inferred_root), names, subdir)
            else:
                print(f"警告: 指定 subdir={subdir!r}，但无法从 yaml 推断 data_root，将仍用 yaml 里的 val 路径")
                data = data_yaml
                print(f"使用 yaml: {data}")
        else:
            data = data_yaml
            print(f"使用 yaml: {data}")
    else:
        raise ValueError("请设置 DATA_ROOT / --data-root，或 DATA_YAML / --data-yaml")

    run_models = models
    if args.model:
        run_models = [("custom", args.model, args.imgsz)]

    for name, path, imgsz in run_models:
        model = YOLO(path)
        metrics = model.val(
            data=data,
            imgsz=args.imgsz if args.model else imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            plots=True,
        )
        print(f"\n===== {name} =====")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall:    {metrics.box.mr:.4f}")
        print(f"mAP50:     {metrics.box.map50:.4f}")
        print(f"mAP50-95:  {metrics.box.map:.4f}")
        print(f"Speed(ms): {metrics.speed}")
