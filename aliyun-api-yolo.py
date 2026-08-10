#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测 + 百炼多模态二次研判（占道晒粮 / 是否在路上 / 是否占半幅路）

用法:
  python aliyun-api-yolo.py --image /path/to/road.jpg
  python aliyun-api-yolo.py --image /path/to/road.jpg --show
  python aliyun-api-yolo.py --image /path/to/road.jpg --no-draw   # 不画图，打印耗时
  python aliyun-api-yolo.py -i a.jpg --manual-box "(782,53)-(1124,784)"  # YOLO 无框时用人工框

Key（二选一）:
  1) 下方 DASHSCOPE_API_KEY = "sk-..."
  2) export DASHSCOPE_API_KEY=sk-...

输出:
  runs/llm_alert/{stem}_boxed.jpg  送大模型的红框图
  runs/llm_alert/{stem}_vis.jpg      研判结果可视化
  runs/llm_alert/{stem}_alert.json   完整 JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import dashscope
import numpy as np
from dashscope import MultiModalConversation
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ===================== 可配置 =====================
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

DASHSCOPE_API_KEY = "sk-7240ae7143814704a164cf6781a2c22a"  # 填 sk-xxx，或留空用环境变量

YOLO_WEIGHTS = "/mnt/sda1/xzm/Code/ultralytics-main/runs/detect/runs/train/ROAD_POTHOLE_GRAIN_DETECT_v11s_1280_0602/weights/best.pt"
YOLO_IMGSZ = 1280
YOLO_CONF = 0.35
YOLO_IOU = 0.45
YOLO_DEVICE = "0"
YOLO_CLASSES_FILTER = ["road-grain"]  # None = 全部类

VL_MODEL = "qwen3-vl-flash"  # 或 qwen3.7-plus，以百炼控制台为准
OUTPUT_DIR = Path("runs/llm_alert")
_YOLO_CACHE: dict[str, YOLO] = {}

FONT_SIZE = 22
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def resolve_font_path() -> str:
    """查找本机可用中文字体."""
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    raise FileNotFoundError(
        "未找到中文字体，请安装: sudo apt install fonts-wqy-microhei fonts-noto-cjk"
    )


def get_font(font_size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_size not in _FONT_CACHE:
        _FONT_CACHE[font_size] = ImageFont.truetype(resolve_font_path(), font_size)
    return _FONT_CACHE[font_size]


def cv2_img_add_text(img_bgr, text, left_top, color_bgr=(0, 255, 0), font_size=FONT_SIZE):
    """在 BGR 图上画中文（PIL，避免 cv2.putText 乱码）."""
    if not text:
        return img_bgr
    x, y = int(left_top[0]), int(left_top[1])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((x, y), str(text), font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def parse_manual_box(text: str) -> tuple[float, float, float, float]:
    """解析人工框，支持 (782,53)-(1124,784) 或 782,53,1124,784."""
    nums = [float(x) for x in re.findall(r"[\d.]+", text.strip())]
    if len(nums) != 4:
        raise ValueError(
            f"manual-box 需要 4 个数字 (x1,y1,x2,y2)，收到: {text!r} -> {nums}"
        )
    x1, y1, x2, y2 = nums
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def detection_from_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    box_id: int = 0,
    class_name: str = "manual",
    conf: float = 1.0,
    source: str = "manual",
) -> dict[str, Any]:
    """由像素坐标构造与 YOLO 一致的 detection 字典."""
    return {
        "box_id": box_id,
        "class_id": -1,
        "class_name": class_name,
        "conf": conf,
        "source": source,
        "xyxy_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "xyxy_norm": [
            round(x1 / img_w, 4),
            round(y1 / img_h, 4),
            round(x2 / img_w, 4),
            round(y2 / img_h, 4),
        ],
    }


def get_api_key() -> str:
    key = (DASHSCOPE_API_KEY or "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "未配置 API Key：设置 DASHSCOPE_API_KEY='sk-...' 或 export DASHSCOPE_API_KEY=sk-..."
        )
    return key


# ===================== YOLO =====================
def run_yolo(
    image_path: str,
    weights: str = YOLO_WEIGHTS,
    imgsz: int = YOLO_IMGSZ,
    conf: float = YOLO_CONF,
    iou: float = YOLO_IOU,
    device: str = YOLO_DEVICE,
    class_filter: list[str] | None = YOLO_CLASSES_FILTER,
) -> tuple[list[dict[str, Any]], int, int, float, float]:
    """返回 detections, h, w, yolo_infer_ms, yolo_load_ms（加载仅首次>0）."""
    yolo_load_ms = 0.0
    if weights not in _YOLO_CACHE:
        t0 = time.perf_counter()
        _YOLO_CACHE[weights] = YOLO(weights)
        yolo_load_ms = (time.perf_counter() - t0) * 1000
    model = _YOLO_CACHE[weights]
    t_pred = time.perf_counter()
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    yolo_infer_ms = (time.perf_counter() - t_pred) * 1000
    r = results[0]
    h, w = r.orig_shape

    allowed_ids = None
    if class_filter:
        allowed_ids = {i for i, n in r.names.items() if n in class_filter}

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls)
        if allowed_ids is not None and cls_id not in allowed_ids:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "box_id": len(detections),
                "class_id": cls_id,
                "class_name": r.names[cls_id],
                "conf": round(float(box.conf), 4),
                "xyxy_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "xyxy_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                "source": "yolo",
            }
        )
    return detections, h, w, yolo_infer_ms, yolo_load_ms


def draw_boxes(image_path: str, detections: list[dict], out_path: Path) -> str:
    """红框图（送大模型）."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy_px"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{d['box_id']}:{d['class_name']} {d['conf']:.2f}"
        img = cv2_img_add_text(img, label, (x1, max(y1 - 28, 0)), (0, 0, 255), font_size=20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path.resolve())


def draw_result_vis(
    image_path: str,
    detections: list[dict],
    llm_judgement: dict | None,
    out_path: Path,
) -> str:
    """研判可视化：绿=告警，黄=不告警，红=仅 YOLO."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    items_by_id = {}
    if llm_judgement and llm_judgement.get("items"):
        for it in llm_judgement["items"]:
            items_by_id[it.get("box_id")] = it

    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy_px"])
        bid = d["box_id"]
        item = items_by_id.get(bid, {})
        alert = item.get("should_alert", False)
        ratio = float(item.get("occupancy_ratio", 0) or 0)
        reason = (item.get("reason") or "")[:36]

        if alert:
            color, tag = (0, 255, 0), "ALERT"
        elif item:
            color, tag = (0, 255, 255), "OK"
        else:
            color, tag = (0, 0, 255), "YOLO"
        if d.get("source") == "manual":
            tag = f"{tag}/人工"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        line1 = f"{bid}:{d['class_name']} {d['conf']:.2f} [{tag}]"
        img = cv2_img_add_text(img, line1, (x1, max(y1 - 30, 2)), color, font_size=20)
        if item:
            line2 = f"占路{ratio:.0%} {reason}"
            img = cv2_img_add_text(
                img, line2, (x1, min(y2 + 4, img.shape[0] - 32)), color, font_size=18
            )

    image_alert = bool(llm_judgement.get("image_alert", False)) if llm_judgement else False
    summary = (llm_judgement or {}).get("summary", "无检测")[:80]
    header = f"告警={image_alert} | {summary}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 48), (0, 0, 0), -1)
    header_color = (0, 255, 0) if image_alert else (255, 255, 255)
    img = cv2_img_add_text(img, header, (8, 8), header_color, font_size=22)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path.resolve())


# ===================== 百炼 LLM =====================
def build_prompt(detections: list[dict], img_w: int, img_h: int) -> str:
    boxes_json = json.dumps(detections, ensure_ascii=False, indent=2)
    return f"""你是道路巡检告警审核员。输入为道路图像（带红色 YOLO 框与编号）及检测坐标。

图像尺寸（像素）: 宽={img_w}, 高={img_h}

【YOLO 检测框】
{boxes_json}

【业务规则 — 占道晒粮】
对每个框判断：
1. target_is_grain_drying: 是否为在路上晾晒粮食/谷物/秸秆（排除阴影、标线、车、绿化、水面反光）。
2. on_carriageway: 是否在机动车通行路面（非纯路肩草、非人行道）。
3. occupancy_ratio: 晒粮占该段车行道宽度的比例 0~1（半幅路约 0.5）。
4. blocks_half_or_more: occupancy_ratio >= 0.25。
5. should_alert: 仅当 1且2且4 均为 true。
6. reason: 一句中文。

【输出】仅 JSON，无 markdown：
{{
  "items": [
    {{
      "box_id": 0,
      "target_is_grain_drying": false,
      "on_carriageway": false,
      "occupancy_ratio": 0.0,
      "blocks_half_or_more": false,
      "should_alert": false,
      "reason": ""
    }}
  ],
  "image_alert": false,
  "summary": "整图结论"
}}"""


def parse_llm_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"回复中未找到 JSON: {text[:800]}")
    return json.loads(text[start : end + 1])


def llm_judge(
    image_path: str,
    detections: list[dict],
    img_w: int,
    img_h: int,
    api_key: str,
    use_boxed_image: bool = True,
    model: str = VL_MODEL,
) -> dict:
    if use_boxed_image and detections:
        boxed = draw_boxes(image_path, detections, OUTPUT_DIR / f"{Path(image_path).stem}_boxed.jpg")
        uri = f"file://{boxed}"
    else:
        uri = f"file://{Path(image_path).resolve()}"

    messages = [
        {
            "role": "user",
            "content": [
                {"image": uri},
                {"text": build_prompt(detections, img_w, img_h)},
            ],
        }
    ]

    response = MultiModalConversation.call(api_key=api_key, model=model, messages=messages)

    if getattr(response, "status_code", 200) != 200:
        raise RuntimeError(f"DashScope 失败: {response}")

    text = response.output.choices[0].message.content[0]["text"]
    return parse_llm_json(text)


# ===================== 流水线 =====================
def pipeline(
    image_path: str,
    weights: str = YOLO_WEIGHTS,
    api_key: str | None = None,
    skip_llm_if_no_det: bool = True,
    model: str = VL_MODEL,
    conf: float = YOLO_CONF,
    draw: bool = True,
    manual_box: str | None = None,
    manual_class: str = "road-grain",
) -> dict:
    image_path = str(Path(image_path).resolve())
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)

    stem = Path(image_path).stem
    api_key = api_key or get_api_key()
    timing: dict[str, float] = {}
    t_total = time.perf_counter()

    detections, h, w, yolo_infer_ms, yolo_load_ms = run_yolo(image_path, weights=weights, conf=conf)
    timing["yolo_infer_ms"] = round(yolo_infer_ms, 1)
    timing["yolo_load_ms"] = round(yolo_load_ms, 1)

    yolo_det_count = len(detections)
    manual_used = False
    if not detections and manual_box:
        x1, y1, x2, y2 = parse_manual_box(manual_box)
        detections = [
            detection_from_xyxy(
                x1, y1, x2, y2, w, h, box_id=0, class_name=manual_class, source="manual"
            )
        ]
        manual_used = True

    result: dict[str, Any] = {
        "image": image_path,
        "image_size": {"width": w, "height": h},
        "yolo_detections": detections,
        "yolo_det_count": yolo_det_count,
        "manual_box_used": manual_used,
        "llm_judgement": None,
        "image_alert": False,
        "boxed_image": None,
        "vis_image": None,
        "timing_ms": timing,
    }

    if not detections and skip_llm_if_no_det:
        result["llm_judgement"] = {
            "items": [],
            "image_alert": False,
            "summary": "YOLO 无目标，未调用大模型",
        }
        timing["llm_ms"] = 0.0
        timing["draw_ms"] = 0.0
        if draw:
            t_d = time.perf_counter()
            result["vis_image"] = draw_result_vis(
                image_path, [], result["llm_judgement"], OUTPUT_DIR / f"{stem}_vis.jpg"
            )
            timing["draw_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        timing["infer_ms"] = round(timing["yolo_infer_ms"] + timing["llm_ms"], 1)
        timing["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
        return result

    t_llm = time.perf_counter()
    llm_out = llm_judge(
        image_path,
        detections,
        w,
        h,
        api_key=api_key,
        model=model,
        use_boxed_image=draw,
    )
    timing["llm_ms"] = round((time.perf_counter() - t_llm) * 1000, 1)

    result["llm_judgement"] = llm_out
    result["image_alert"] = bool(llm_out.get("image_alert", False))
    timing["draw_ms"] = 0.0
    if draw:
        t_d = time.perf_counter()
        if detections:
            result["boxed_image"] = str((OUTPUT_DIR / f"{stem}_boxed.jpg").resolve())
        result["vis_image"] = draw_result_vis(
            image_path, detections, llm_out, OUTPUT_DIR / f"{stem}_vis.jpg"
        )
        timing["draw_ms"] = round((time.perf_counter() - t_d) * 1000, 1)

    timing["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
    timing["infer_ms"] = round(timing["yolo_infer_ms"] + timing["llm_ms"], 1)
    return result


def main():
    parser = argparse.ArgumentParser(description="YOLO + 百炼 占道晒粮告警（含可视化）")
    parser.add_argument("--image", "-i", required=True, help="原图路径")
    parser.add_argument("--weights", "-w", default=YOLO_WEIGHTS)
    parser.add_argument("--model", default=VL_MODEL)
    parser.add_argument("--conf", type=float, default=YOLO_CONF)
    parser.add_argument("--out", default=None, help="JSON 保存路径")
    parser.add_argument("--show", action="store_true", help="弹窗显示 vis 图")
    parser.add_argument("--no-draw", action="store_true", help="不画图，仅检测+大模型（测速用）")
    parser.add_argument(
        "--manual-box",
        default=None,
        help='YOLO 无框时使用人工框，如 "(782,53)-(1124,784)" 或 "782,53,1124,784"',
    )
    parser.add_argument(
        "--manual-class",
        default="road-grain",
        help="人工框在送审/可视化中的类别名，默认 road-grain",
    )
    args = parser.parse_args()

    out = pipeline(
        args.image,
        weights=args.weights,
        model=args.model,
        conf=args.conf,
        draw=not args.no_draw,
        manual_box=args.manual_box,
        manual_class=args.manual_class,
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))

    t = out.get("timing_ms", {})
    if t:
        print("\n===== 耗时(s) =====")
        print(f"  YOLO 推理:     {t.get('yolo_infer_ms', 0) / 1000:.2f}")
        if t.get("yolo_load_ms", 0) > 0:
            print(f"  YOLO 加载权重: {t['yolo_load_ms'] / 1000:.2f}  (仅本次进程首次)")
        print(f"  大模型 API:    {t.get('llm_ms', 0) / 1000:.2f}")
        print(f"  画图:          {t.get('draw_ms', 0) / 1000:.2f}")
        print(f"  推理合计:      {t.get('infer_ms', 0) / 1000:.2f}  (YOLO+LLM，不含画图)")
        print(f"  总耗时:        {t.get('total_ms', 0) / 1000:.2f}")

    save = Path(args.out) if args.out else OUTPUT_DIR / f"{Path(args.image).stem}_alert.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if out.get("manual_box_used"):
        print("人工框: 已启用（YOLO 未检出，使用 --manual-box）")
    print(f"\nJSON: {save.resolve()}")
    if out.get("boxed_image"):
        print(f"boxed: {out['boxed_image']}")
    if out.get("vis_image"):
        print(f"vis:   {out['vis_image']}")
    print("\n>>> 需要告警 <<<" if out["image_alert"] else "\n>>> 无需告警 <<<")

    if args.show and out.get("vis_image"):
        vis = cv2.imread(out["vis_image"])
        if vis is not None:
            cv2.imshow("aliyun-api-yolo", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()