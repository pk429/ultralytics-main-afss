"""鼠标画框 + 撤回"""

import sys
import cv2
import numpy as np

# ========== 配置 ==========
IMAGE_PATH = "/home/lenovo/桌面/test/img_v3_0212v_0c383f63-4f85-41d9-a797-3af460b9844g.jpg"  


class BoxDrawer:
    def __init__(self, image):
        self.base = image.copy()
        self.boxes = []          # 当前所有框
        self.history = []        # 撤回栈：每次确定框前保存快照

        self.drawing = False
        self.start = (0, 0)
        self.end = (0, 0)

    def snapshot(self):
        self.history.append(self.boxes.copy())

    def undo(self):
        if self.history:
            self.boxes = self.history.pop()
            print(f"已撤回，当前框数量: {len(self.boxes)}")
        else:
            print("没有可撤回的操作")

    def clear(self):
        if self.boxes:
            self.snapshot()
            self.boxes.clear()
            print("已清空所有框")
        else:
            print("当前没有框")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)

            x1, y1 = self.start
            x2, y2 = self.end
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if x2 - x1 > 2 and y2 - y1 > 2:
                self.snapshot()  # 先保存，再添加，便于撤回
                self.boxes.append((x1, y1, x2, y2))
                w, h = x2 - x1, y2 - y1
                print(f"框 #{len(self.boxes)}: ({x1},{y1})-({x2},{y2})  宽={w}px  高={h}px")

    def norm_rect(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def render(self):
        img = self.base.copy()

        for i, (x1, y1, x2, y2) in enumerate(self.boxes, 1):
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"#{i} {w}x{h}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.drawing:
            x1, y1, x2, y2 = self.norm_rect(self.start, self.end)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{w}x{h}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 底部提示
        tip = "Drag: draw | u / Ctrl+Z: undo | c: clear | s: save | q: quit"
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, tip, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return img


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取: {path}")
        return

    drawer = BoxDrawer(img)
    win = "Box Drawer"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, drawer.on_mouse)

    while True:
        cv2.imshow(win, drawer.render())
        key = cv2.waitKey(20) & 0xFF

        # u 或 Ctrl+Z 撤回
        if key in (ord("u"), ord("U"), 26):
            drawer.undo()
        elif key in (ord("c"), ord("C")):
            drawer.clear()
        elif key in (ord("s"), ord("S")):
            cv2.imwrite("annotated.jpg", drawer.render())
            print("已保存 annotated.jpg")
        elif key in (ord("q"), ord("Q"), 27):  # q 或 Esc
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()