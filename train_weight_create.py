from pathlib import Path

old_img_dir = Path("/mnt/sda1/xzm/datasets/cargoship_visible_obb_dataset/classifty/split/images/train")
new_img_dir = Path("/mnt/sda1/xzm/datasets/cargoship_visible_obb_dataset/classifty/cargoship_dataset_0727/split/images/train/")
out_txt = Path("/mnt/sda1/xzm/datasets/cargoship_visible_obb_dataset/classifty/train_weighted.txt")

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

new_repeat = 5  # 新数据重复 5 倍，可改 3-10

old_imgs = sorted(p for p in old_img_dir.rglob("*") if p.suffix.lower() in img_exts)
new_imgs = sorted(p for p in new_img_dir.rglob("*") if p.suffix.lower() in img_exts)

lines = []
lines += [str(p) for p in old_imgs]
for _ in range(new_repeat):
    lines += [str(p) for p in new_imgs]

out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"old: {len(old_imgs)}")
print(f"new: {len(new_imgs)} x {new_repeat}")
print(f"total samples per epoch: {len(lines)}")
print(f"saved: {out_txt}")