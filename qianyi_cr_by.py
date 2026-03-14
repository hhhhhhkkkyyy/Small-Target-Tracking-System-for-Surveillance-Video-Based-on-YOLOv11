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

    # ===================== 3. 加载 CrowdHuman 普通 baseline 权重 =====================
    CROWDHUMAN_BASELINE_WEIGHT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\crowdhuman_yolov11s_baseline\weights\best.pt"

    if not os.path.exists(CROWDHUMAN_BASELINE_WEIGHT):
        raise FileNotFoundError(
            f"未找到 CrowdHuman baseline 权重：{CROWDHUMAN_BASELINE_WEIGHT}\n请先完成 base_tr.py 训练！")

    print(f"加载 CrowdHuman 普通预训练权重：{CROWDHUMAN_BASELINE_WEIGHT}")
    model = YOLO(CROWDHUMAN_BASELINE_WEIGHT)

    # ===================== 4. 迁移训练到 MOT17（控制精度略高于纯COCO） =====================
    model.train(
        data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\data.yaml",

        epochs=50,  # 比纯COCO多一点轮数，但不充分
        imgsz=800,  # 比640大，但比1280小（控制小目标精度）
        batch=4,  # 保持适中
        device=0,

        # 迁移学习关键：学习率要小
        lr0=8e-5,  # 很小，避免破坏 CrowdHuman 学到的特征
        lrf=0.01,  # 最终学习率也很小
        warmup_epochs=3,
        weight_decay=5e-4,  # 正常值

        # 数据增强：适度开启，但不强
        mosaic=0.3,  # 轻度 mosaic（比0强，但远低于1.0）
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,  # 正常水平翻转
        hsv_h=0.015,
        hsv_s=0.4,  # 饱和度增强减弱
        hsv_v=0.3,  # 亮度增强减弱

        # 损失权重：稍微偏向检测框
        box=7.5,  # 略低于默认的7.5~10.0
        cls=0.6,  # 单类任务适当降低
        dfl=1.5,

        # 其他
        amp=True,
        single_cls=True,
        patience=15,  # 比纯COCO稍宽松
        optimizer="AdamW",  # 保持较好的优化器

        # 输出命名（方便区分）
        project="runs/detect",
        name="mot17_yolov11s_cr_by",
        exist_ok=True,
        save=True,
        val=True,
        plots=True
    )

    print("\n训练完成！")
