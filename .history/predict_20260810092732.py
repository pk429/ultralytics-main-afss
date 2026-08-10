import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO(model='/mnt/sda1/xzm/Code/ultralytics-main/runs/detect/runs/train/FISHER_DETECT_v11n_p2_1920_0527/weights/best.pt')
    # model = YOLO(model='/mnt/sda1/xzm/Code/ultralytics-main/runs/detect/runs/train/FISHER_DETECT_v11s_p2_1280_0529/weights/best.pt')
    # model = YOLO(model='/mnt/sda1/xzm/Code/ultralytics-main/runs/detect/runs/train/ROAD_POTHOLE_GRAIN_DETECT_v11s_1280_0608/weights/best.pt')

    # # model = YOLO(model='/home/lenovo/下载/model_fish_1280v11s.pt')
    # model.predict(
    #     source=r'/mnt/sda1/xzm/video/grain/飞书20260604-113341.mp4',  # 视频路径
    #     imgsz=[1280,1280],
    #     save=True,          # 保存结果视频
    #     show=True,         # 实时显示窗口
    #     conf=0.25,           # 置信度阈值
    #     iou=0.5,           # NMS IOU阈值
    #     device='0',      # 用GPU推理，改成 'cpu' 则用CPU
    #     save_txt=False,      # 保存检测结果为txt
    #     save_conf=False,     # txt里包含置信度
    #     line_width=2,       # 边框线宽
    #     # project=r'./output.mp4',  # 结果保存目录
    #     name='./result',      # 结果子文件夹名
    # )

    # 钓鱼识别
    model = YOLO(
        model=" /mnt/sda1/xzm/Code/ultralytics-main-afss/runs/detect/runs/train/SWIMMER_DETECT_v11s_1280_AFSS_0809/weights/best.pt"
    )
    results = model.predict(
        source=r"/mnt/sda1/xzm/datasets/swimmer/data/20260807_test_dataset/images/",  # 视频路径
        # source=r'/mnt/sda1/xzm/datasets/fisher/test_datasets/20260616_test_dataset/images/',  # 图片目录
        imgsz=1280,
        save=True,  # 保存结果视频
        show=False,  # 实时显示窗口
        conf=0.5,  # 置信度阈值
        iou=0.45,  # NMS IOU阈值
        device="0",  # 用GPU推理，改成 'cpu' 则用CPU
        save_txt=True,  # 保存检测结果为txt
        save_conf=False,  # txt里包含置信度
        line_width=2,  # 边框线宽
        project=r"",  # 结果保存目录
        name="",  # 结果子文件夹名
        stream=True,  # 长视频必须流式处理，避免结果累积占满内存
    )
    for _ in results:
        pass
