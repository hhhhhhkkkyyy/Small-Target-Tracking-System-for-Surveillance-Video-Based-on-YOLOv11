# import os
# from tqdm import tqdm
# import shutil
#
# # ====================== 配置 ======================
# ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\crowdhuman"  # 你的crowdhuman根目录
# MIN_AREA = 0.0005  # 过滤极小目标（面积占比 < 0.05%，可选，CrowdHuman基本不需要）
#
#
# # ====================== 检查 + 过滤函数 ======================
# def check_and_clean(split):
#     img_dir = os.path.join(ROOT, split, "images")
#     label_dir = os.path.join(ROOT, split, "labels")
#
#     valid_count = 0
#     removed = 0
#
#     for label_file in tqdm(os.listdir(label_dir), desc=f"检查 {split} labels"):
#         label_path = os.path.join(label_dir, label_file)
#         img_path = os.path.join(img_dir, label_file.replace(".txt", ".jpg"))
#
#         if not os.path.exists(img_path):
#             os.remove(label_path)
#             removed += 1
#             continue
#
#         with open(label_path, "r") as f:
#             lines = f.readlines()
#
#         new_lines = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             cls, cx, cy, w, h = map(float, parts)
#             area = w * h
#             if area < MIN_AREA:  # 过滤极小（CrowdHuman很少，可设0不过滤）
#                 continue
#             new_lines.append(line.strip())
#
#         # 写回
#         with open(label_path, "w") as f:
#             f.write("\n".join(new_lines))
#
#         if len(new_lines) > 0 or MIN_AREA == 0:  # 保留有目标或不过滤的
#             valid_count += 1
#
#     print(f"{split} 完成：有效图像 {valid_count}，移除标签 {removed}")
#
#
# # ====================== 主程序 ======================
# if __name__ == "__main__":
#     # 检查 train / valid / test
#     check_and_clean("train")
#     check_and_clean("valid")
#     check_and_clean("test")
#
#     print("\n✅ CrowdHuman 预处理完成！可以直接训练了！")
#     print("接下来用下面脚本训练基线：")
#
#     # 顺便打印训练命令示例
#     print("""
# from ultralytics import YOLO
# model = YOLO("yolov11n.pt")
# model.train(data="crowdhuman/data.yaml", epochs=50, imgsz=800, batch=16, name="crowdhuman_baseline")
#     """)

#
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
#
#
# def preprocess_small_target_images(img_root, target_size=(800, 800), save_enhanced=True):
#     """
#     小目标图片静态预处理：
#     1. 统一尺寸（等比例缩放+letterbox填充，避免拉伸）
#     2. 自适应对比度增强（CLAHE），凸显小目标特征
#     3. 可选：保存增强后的图片（覆盖/另存）
#
#     Args:
#         img_root: 图片根目录（如MOT17_yolo/images/train）
#         target_size: 目标尺寸 (w, h)，建议800×800/1280×720
#         save_enhanced: 是否保存增强后的图片（True=覆盖原图，False=另存为enh_xxx.jpg）
#     """
#     # 遍历所有图片文件
#     img_ext = [".jpg", ".jpeg", ".png", ".bmp"]
#     img_files = []
#     for root, dirs, files in os.walk(img_root):
#         for file in files:
#             if os.path.splitext(file)[-1].lower() in img_ext:
#                 img_files.append(os.path.join(root, file))
#
#     # 批量处理图片
#     for img_path in tqdm(img_files, desc="小目标图片预处理"):
#         # 1. 读取图片（BGR格式）
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"⚠️ 跳过无效图片：{img_path}")
#             continue
#         h, w = img.shape[:2]
#
#         # 2. 尺寸统一：等比例缩放 + letterbox填充（避免小目标拉伸）
#         scale = min(target_size[0] / w, target_size[1] / h)  # 计算缩放比例（取最小，避免超出）
#         new_w, new_h = int(w * scale), int(h * scale)
#         img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)  # 立方插值，保留小目标细节
#
#         # 创建目标画布（灰色填充，避免黑边干扰小目标）
#         canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 128  # 128=灰色背景
#         x_offset = (target_size[0] - new_w) // 2  # 水平居中
#         y_offset = (target_size[1] - new_h) // 2  # 垂直居中
#         canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
#
#         # 3. 小目标对比度增强：CLAHE（自适应直方图均衡化）
#         # 转换为LAB色彩空间（亮度通道单独增强，避免色彩失真）
#         lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
#         l_channel, a_channel, b_channel = cv2.split(lab)
#
#         # CLAHE增强亮度通道（提升小目标与背景的对比度）
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # clipLimit=2.0：增强幅度，适合小目标
#         l_channel_enhanced = clahe.apply(l_channel)
#
#         # 合并通道，转回BGR
#         lab_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
#         img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
#
#         # 4. 保存增强后的图片
#         if save_enhanced:
#             # 覆盖原始图片（推荐，减少磁盘占用）
#             cv2.imwrite(img_path, img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 95=高质量保存
#         else:
#             # 另存为新文件（避免覆盖原图，方便对比）
#             img_name = os.path.basename(img_path)
#             save_path = os.path.join(os.path.dirname(img_path), f"enh_{img_name}")
#             cv2.imwrite(save_path, img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
#
#
# # ===================== 调用示例（替换为你的路径） =====================
# if __name__ == "__main__":
#     # MOT17训练集图片路径
#     mot17_train_img = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\images\train"
#     mot17_val_img = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\images\val"
#
#     # CrowdHuman数据集路径（可选）
#     # crowdhuman_train_img = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\crowdhuman\train\images"
#
#     # 预处理（目标尺寸800×800，覆盖原图）
#     preprocess_small_target_images(mot17_train_img, target_size=(800, 800), save_enhanced=True)
#     preprocess_small_target_images(mot17_val_img, target_size=(800, 800), save_enhanced=True)
#     # preprocess_small_target_images(crowdhuman_train_img, target_size=(800, 800), save_enhanced=True)
#
#     print("✅ 静态阶段图片预处理完成：尺寸统一+对比度增强")
import os
import cv2
import shutil
import random
from pathlib import Path

# ===================== 配置区 =====================
MOT_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17"
SAVE_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo"
VAL_RATIO = 0.1  # 训练集按10%比例划分验证集
CLASS_ID = 0  # person类别ID
DETECTOR_FILTER = "FRCNN"  # 只处理FRCNN检测器的序列
random.seed(42)  # 固定随机种子确保结果可复现


# =================================================


def convert_bbox(img_w, img_h, x, y, w, h):
    """将MOT格式的bbox转换为YOLO格式（归一化中心坐标+宽高）"""
    cx = x + w / 2
    cy = y + h / 2
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def process_sequence(seq_path, save_dirs, split_type):
    """处理单个序列数据"""
    img_dir = seq_path / "img1"
    gt_file = seq_path / "gt" / "gt.txt"

    # 检查图片目录是否存在且有图片
    img_files = list(img_dir.glob("*.jpg"))
    if not img_files:
        print(f"⚠️ {seq_path.name} 没有找到图片文件，跳过")
        return

    # 获取图片尺寸
    sample_img = cv2.imread(str(img_files[0]))
    img_h, img_w = sample_img.shape[:2]

    # 解析GT文件，按帧存储bbox
    frame_dict = {}
    if gt_file.exists():
        with open(gt_file, "r") as f:
            for line in f:
                items = line.strip().split(",")
                # 跳过不完整的行
                if len(items) < 8:
                    continue

                frame_id = int(items[0])
                conf = int(items[6]) if len(items) > 6 else 1
                cls = int(items[7]) if len(items) > 7 else 1

                # 只保留置信度为1且类别为1（行人）的标注
                if conf != 1 or cls != 1:
                    continue

                # 解析bbox坐标
                try:
                    x, y, w, h = map(float, items[2:6])
                except ValueError:
                    continue

                # 过滤无效bbox
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    continue

                # 转换为YOLO格式
                box = convert_bbox(img_w, img_h, x, y, w, h)
                frame_dict.setdefault(frame_id, []).append(box)
    else:
        print(f"⚠️ {seq_path.name} 没有找到GT文件，仅复制图片")

    # 处理每张图片
    for img_path in img_files:
        frame_id = int(img_path.stem)

        # 确定保存目录
        if split_type == "test":
            # 测试集单独存放
            img_save = save_dirs["img_test"]
            lbl_save = save_dirs["lbl_test"]
        else:
            # 训练集按比例划分训练/验证
            if random.random() < VAL_RATIO:
                img_save = save_dirs["img_val"]
                lbl_save = save_dirs["lbl_val"]
            else:
                img_save = save_dirs["img_train"]
                lbl_save = save_dirs["lbl_train"]

        # 复制图片
        shutil.copy(img_path, img_save / img_path.name)

        # 生成标签文件
        label_path = lbl_save / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            for box in frame_dict.get(frame_id, []):
                f.write(f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in box) + "\n")


def main():
    # 创建保存目录
    save_dirs = {
        "img_train": Path(SAVE_ROOT) / "images/train",
        "lbl_train": Path(SAVE_ROOT) / "labels/train",
        "img_val": Path(SAVE_ROOT) / "images/val",
        "lbl_val": Path(SAVE_ROOT) / "labels/val",
        "img_test": Path(SAVE_ROOT) / "images/test",
        "lbl_test": Path(SAVE_ROOT) / "labels/test",
    }

    # 创建所有目录
    for p in save_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    # 统计处理的序列数量
    processed_seqs = 0

    # 处理所有split（train和test）
    for split_type in ["train", "test"]:
        split_path = Path(MOT_ROOT) / split_type
        if not split_path.exists():
            print(f"⚠️ 未找到 {split_type} 目录，跳过")
            continue

        # 只获取包含FRCNN检测器的序列目录
        seq_dirs = [
            d for d in split_path.iterdir()
            if d.is_dir() and "MOT17-" in d.name and DETECTOR_FILTER in d.name
        ]

        if not seq_dirs:
            print(f"⚠️ {split_type} 目录下没有找到{DETECTOR_FILTER}检测器的序列文件夹")
            continue

        # 处理每个FRCNN序列
        for seq in seq_dirs:
            print(f"📄 处理 {split_type}/{seq.name}")
            process_sequence(seq, save_dirs, split_type)
            processed_seqs += 1

    # 修复data.yaml生成逻辑（修正语法错误）
    save_root_unix = str(Path(SAVE_ROOT)).replace("\\", "/")
    yaml_content = f"""# MOT17 YOLO格式数据集配置（仅FRCNN序列）
path: {save_root_unix}
train: images/train  # 训练集图片路径
val: images/val      # 验证集图片路径
test: images/test    # 测试集图片路径

# 类别配置
nc: 1  # 类别数
names: ['person']  # 类别名称
"""

    # 保存data.yaml文件
    yaml_path = Path(SAVE_ROOT) / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    print("\n✅ MOT17 数据集转换完成（仅FRCNN检测器序列）！")
    print(f"📊 共处理 {processed_seqs} 个序列")
    print(f"📁 保存路径: {SAVE_ROOT}")
    print(f"📝 生成配置文件: {yaml_path}")


if __name__ == "__main__":
    main()