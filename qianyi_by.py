from ultralytics import YOLO
import os
import numpy as np


# ===================== 1. 修复 MOT17 YOLO 标注（保持不变） =====================
def fix_mot17_labels(label_dir):
    """
    修正 MOT17 → YOLO 格式标注：
    1. 裁剪坐标到 [0,1]
    2. 移除非法 / 负值框
    3. 保证 w,h > 0
    """
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for file in label_files:
        file_path = os.path.join(label_dir, file)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])

                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, 1.0)
                w = np.clip(w, 0.0, 1.0 - x)
                h = np.clip(h, 0.0, 1.0 - y)

                if w > 0 and h > 0:
                    fixed_lines.append(
                        f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    )
            except:
                continue

        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)


# ===================== 2. 主入口 =====================
if __name__ == '__main__':

    # ------------------ 数据集路径 ------------------
    train_label_dir = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\train"
    val_label_dir = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\val"

    if os.path.exists(train_label_dir):
        fix_mot17_labels(train_label_dir)
        print("✅ 训练集标注修正完成")

    if os.path.exists(val_label_dir):
        fix_mot17_labels(val_label_dir)
        print("✅ 验证集标注修正完成")

    # ------------------ 3. 加载官方预训练权重 ------------------
    PATH = r'D:\666\PyCharm 2023.3.2\pythonProject\dadada\yolo11s.pt'
    model = YOLO(PATH)  # 自动下载或使用本地文件

    # ------------------ 4. 最纯净的训练调用（只改必要参数） ------------------
    model.train(
        data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\data.yaml",

        # 只保留你指定的三个参数
        epochs=50,
        imgsz=640,
        batch=4,

        # 以下这些是比较重要的显式设置（建议保留），其他都用官方默认
        device=0,  # 使用 GPU 0
        single_cls=True,  # MOT17 只有行人一类，强烈建议保留
        name="mot17_yolov11s_by",  # 好区分的实验名

        # 下面这些行可以完全删除，让它们用官方默认值：
        # optimizer, lr0, momentum, weight_decay, warmup_epochs,
        # mosaic, mixup, copy_paste, fliplr, hsv_*, amp, patience, etc.
    )
