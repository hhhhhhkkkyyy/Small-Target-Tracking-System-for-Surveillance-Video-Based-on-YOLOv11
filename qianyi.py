# from ultralytics import YOLO
# import os
# import numpy as np
#
#
# # 修复MOT17标注文件的函数（处理坐标越界/负值问题）
# def fix_mot17_labels(label_dir):
#     """
#     修正MOT17数据集标注文件：
#     1. 裁剪坐标到[0,1]范围
#     2. 移除负坐标标注
#     3. 确保宽高为正
#     """
#     label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
#     for file in label_files:
#         file_path = os.path.join(label_dir, file)
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
#
#         fixed_lines = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue  # 跳过无效行
#
#             # MOT17标注格式：class x y w h (已归一化)
#             try:
#                 cls = int(parts[0])
#                 x, y, w, h = map(float, parts[1:5])
#
#                 # 修正坐标：裁剪到[0,1]范围
#                 x = np.clip(x, 0.0, 1.0)
#                 y = np.clip(y, 0.0, 1.0)
#                 w = np.clip(w, 0.0, 1.0 - x)  # 宽度不能超出图片
#                 h = np.clip(h, 0.0, 1.0 - y)  # 高度不能超出图片
#
#                 # 过滤无效标注（宽高为0）
#                 if w > 0 and h > 0:
#                     fixed_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
#             except:
#                 continue  # 跳过格式错误行
#
#         # 保存修正后的标注
#         with open(file_path, 'w') as f:
#             f.writelines(fixed_lines)
#
#
# if __name__ == '__main__':  # Windows多进程必须的封装
#     # -------------------------- 1. 数据集预处理 --------------------------
#     # 替换为你的MOT17标签目录
#     train_label_dir = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\train"
#     val_label_dir = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\val"
#
#     # 修正训练集和验证集标注
#     if os.path.exists(train_label_dir):
#         fix_mot17_labels(train_label_dir)
#         print("训练集标注修正完成")
#     if os.path.exists(val_label_dir):
#         fix_mot17_labels(val_label_dir)
#         print("验证集标注修正完成")
#
#     # -------------------------- 2. 迁移训练配置 --------------------------
#     # 加载预训练模型
#     PATH = r'D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\crowdhuman_yolov11s_gam_siou1\weights\best.pt'
#     model = YOLO(PATH)
#
#     # 迁移训练（针对MOT17行人检测优化）
#     results = model.train(
#         # 数据集配置
#         data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\data.yaml",
#         epochs=50,  # 迁移训练epoch数（建议20-50）
#         imgsz=1280,  # 输入图片尺寸
#         batch=4,  # 批次大小（根据GPU显存调整）
#         device=0,  # 使用第0块GPU
#
#         # 学习率配置（迁移学习关键）
#         lr0=1e-4,  # 初始学习率（预训练模型建议1e-4 ~ 1e-3）
#         lrf=1e-2,  # 最终学习率因子
#         warmup_epochs=2,  # 预热epoch
#         weight_decay=0.0005,  # 权重衰减
#
#         # 数据增强（MOT17视频数据适配）
#         mosaic=0.0,  # 关闭mosaic（视频帧连续性）
#         mixup=0.0,  # 关闭mixup
#         copy_paste=0.0,  # 关闭copy_paste
#         fliplr=0.5,  # 水平翻转（行人检测常用）
#         hsv_h=0.015,  # 色调增强
#         hsv_s=0.7,  # 饱和度增强
#         hsv_v=0.4,  # 明度增强
#
#         # 训练策略
#         close_mosaic=0,  # 全程关闭mosaic
#         amp=True,  # 混合精度训练
#         single_cls=True,  # MOT17只有行人一类
#         patience=10,  # 早停耐心值（可选）
#
#         # 输出配置
#         project="runs/detect",
#         name="mot17_final",
#         exist_ok=True,  # 覆盖已有实验
#         save=True,  # 保存权重
#         val=True  # 训练中验证
#     )
from ultralytics import YOLO
import os
import numpy as np

# 修复MOT17标注文件的函数（处理坐标越界/负值问题，保留原功能）
def fix_mot17_labels(label_dir):
    """
    修正MOT17数据集标注文件：
    1. 裁剪坐标到[0,1]范围
    2. 移除负坐标标注
    3. 确保宽高为正
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
                continue  # 跳过无效行

            # MOT17标注格式：class x y w h (已归一化)
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])

                # 修正坐标：裁剪到[0,1]范围
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, 1.0)
                w = np.clip(w, 0.0, 1.0 - x)  # 宽度不能超出图片
                h = np.clip(h, 0.0, 1.0 - y)  # 高度不能超出图片

                # 过滤无效标注（宽高为0）
                if w > 0 and h > 0:
                    fixed_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            except:
                continue  # 跳过格式错误行

        # 保存修正后的标注
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)


if __name__ == '__main__':  # Windows多进程必须的封装
    # -------------------------- 1. 可选：再次校验标注（非必需，可注释） --------------------------
    train_label_dir = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\train"
    val_label_dir = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\val"

    # 若之前已修正，可直接注释以下两段
    if os.path.exists(train_label_dir):
        fix_mot17_labels(train_label_dir)
        print("训练集标注二次校验完成")
    if os.path.exists(val_label_dir):
        fix_mot17_labels(val_label_dir)
        print("验证集标注二次校验完成")

    # -------------------------- 2. 续训配置：加载MOT17迁移训练后的best.pt --------------------------
    # 替换为你第一次迁移训练得到的best.pt路径（关键：加载续训权重）
    CONTINUE_TRAIN_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_1_baseline\weights\best.pt"
    model = YOLO(CONTINUE_TRAIN_PATH)  # 加载已有最优权重，继续训练

    # -------------------------- 3. 续训参数优化（核心：适配续训场景，进一步提升性能） --------------------------
    results = model.train(
        # 数据集配置（保持不变，与第一次迁移训练一致）
        data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\data.yaml",
        epochs=50,  # 续训50轮（按需调整，建议30-50轮，避免过拟合）
        imgsz=1280,  # 保持输入尺寸一致，不随意修改
        batch=4,  # 可根据GPU显存微调（如显存充足可改为8，需保持稳定）
        device=0,  # 保持GPU设备一致

        # 学习率配置（续训关键：降低初始学习率，避免震荡）
        lr0=5e-5,  # 续训初始学习率（比第一次迁移训练更低，1e-5 ~ 5e-5最优）
        lrf=1e-3,  # 最终学习率因子（同步降低，增强训练稳定性）
        warmup_epochs=1,  # 续训预热轮次减少（仅1轮，快速进入稳定训练）
        weight_decay=0.001,  # 适度提高权重衰减，抑制过拟合（续训易过拟合，关键优化）

        # 数据增强（微调：适度增强，平衡泛化能力与拟合效果）
        mosaic=0.0,  # 仍关闭mosaic，保持视频数据连续性
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.3,  # 适度降低水平翻转概率，减少冗余增强
        hsv_h=0.01,  # 降低色调增强幅度，避免偏离真实场景
        hsv_s=0.5,   # 降低饱和度增强幅度，提升稳定性
        hsv_v=0.3,   # 降低明度增强幅度，减少噪声

        # 训练策略（续训优化：强化拟合，防止过拟合）
        close_mosaic=0,
        amp=True,  # 保持混合精度训练，提升效率
        single_cls=True,
        patience=15,  # 提高早停耐心值，给模型更多收敛时间（续训收敛更慢）
        cos_lr=True,  # 启用余弦学习率调度（续训时收敛更平滑，关键优化）

        # 输出配置（区分第一次训练，便于查找结果）
        project="runs/detect",
        name="mot17_final_continue",  # 自定义续训实验名称，避免覆盖原结果
        exist_ok=True,
        save=True,  # 保存续训后的最优权重
        val=True,   # 训练中验证，实时监控性能
        save_period=10,  # 每10轮保存一次权重，便于回溯最优模型
        plots=True  # 生成续训训练曲线，便于对比分析
    )
# from ultralytics import YOLO
# import os
# import numpy as np
#
#
# # ===================== 1. 修复 MOT17 YOLO 标注 =====================
# def fix_mot17_labels(label_dir):
#     """
#     修正 MOT17 → YOLO 格式标注：
#     1. 裁剪坐标到 [0,1]
#     2. 移除非法 / 负值框
#     3. 保证 w,h > 0
#     """
#     label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
#
#     for file in label_files:
#         file_path = os.path.join(label_dir, file)
#
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
#
#         fixed_lines = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#
#             try:
#                 cls = int(parts[0])
#                 x, y, w, h = map(float, parts[1:5])
#
#                 # 裁剪到合法范围
#                 x = np.clip(x, 0.0, 1.0)
#                 y = np.clip(y, 0.0, 1.0)
#                 w = np.clip(w, 0.0, 1.0 - x)
#                 h = np.clip(h, 0.0, 1.0 - y)
#
#                 if w > 0 and h > 0:
#                     fixed_lines.append(
#                         f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
#                     )
#             except:
#                 continue
#
#         with open(file_path, 'w') as f:
#             f.writelines(fixed_lines)
#
#
# # ===================== 2. 主入口 =====================
# if __name__ == '__main__':
#
#     # ------------------ 数据集路径 ------------------
#     train_label_dir = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\train"
#     val_label_dir   = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\labels\val"
#
#     if os.path.exists(train_label_dir):
#         fix_mot17_labels(train_label_dir)
#         print("✅ 训练集标注修正完成")
#
#     if os.path.exists(val_label_dir):
#         fix_mot17_labels(val_label_dir)
#         print("✅ 验证集标注修正完成")
#
#     # ------------------ 3. 使用 YOLOv11s 官方权重 ------------------
#     # ❗ 不使用 CrowdHuman，不迁移
#     PATH = r'D:\666\PyCharm 2023.3.2\pythonProject\dadada\yolo11s.pt'
#     model = YOLO(PATH)
#
#     # ------------------ 4. MOT17 训练 ------------------
#     model.train(
#         data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\data.yaml",
#         epochs=50,
#         imgsz=1280,
#         batch=4,
#         device=0,
#
#         # 学习率（从 COCO 预训练开始，稍大一点是合理的）
#         lr0=1e-3,
#         lrf=1e-2,
#         warmup_epochs=3,
#         weight_decay=5e-4,
#
#         # 视频数据增强策略（MOT 推荐）
#         mosaic=0.0,
#         mixup=0.0,
#         copy_paste=0.0,
#         fliplr=0.5,
#         hsv_h=0.015,
#         hsv_s=0.7,
#         hsv_v=0.4,
#         close_mosaic=0,
#
#         # 训练设置
#         amp=True,
#         single_cls=True,
#         patience=10,
#
#         # 输出
#         project="runs/detect",
#         name="mot17_yolov11s_from_scratch1",
#         exist_ok=True,
#         save=True,
#         val=True
#     )
