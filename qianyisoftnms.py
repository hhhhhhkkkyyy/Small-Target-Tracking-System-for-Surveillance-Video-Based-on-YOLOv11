import torch
import warnings
import os
from ultralytics import YOLO

# 解决 DetectionPredictor 导入爆红的核心：兼容不同版本的路径
try:
    # 新版本（Ultralytics 8.0+ / YOLOv11 最新版）
    from ultralytics.models.yolo.detect import DetectionPredictor
except ImportError:
    try:
        # 旧版本（早期 YOLOv11 / YOLOv8）
        from ultralytics.yolo.detect.predict import DetectionPredictor
    except ImportError:
        # 终极兼容：从 ultralytics 核心模块导入
        from ultralytics.engine.predictor import BasePredictor


        # 自定义 DetectionPredictor 基类（兜底方案）
        class DetectionPredictor(BasePredictor):
            pass

warnings.filterwarnings("ignore")


# ===================== 1. Soft-NMS 核心实现 =====================
def soft_nms_pytorch(
        boxes,
        scores,
        sigma=0.5,
        score_thresh=0.001,
        method="gaussian"
):
    if len(boxes) == 0:
        return torch.tensor([], device=boxes.device)

    idxs = scores.argsort(descending=True)
    keep = []

    while idxs.numel() > 0:
        top_idx = idxs[0]
        keep.append(top_idx)

        if idxs.numel() == 1:
            break

        top_box = boxes[top_idx].unsqueeze(0)
        rest_boxes = boxes[idxs[1:]]

        xx1 = torch.max(top_box[:, 0], rest_boxes[:, 0])
        yy1 = torch.max(top_box[:, 1], rest_boxes[:, 1])
        xx2 = torch.min(top_box[:, 2], rest_boxes[:, 2])
        yy2 = torch.min(top_box[:, 3], rest_boxes[:, 3])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_top = (top_box[:, 2] - top_box[:, 0]) * (top_box[:, 3] - top_box[:, 1])
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        iou = inter / (area_top + area_rest - inter + 1e-6)

        if method == "gaussian":
            weight = torch.exp(-(iou ** 2) / sigma)
        else:
            weight = torch.ones_like(iou)
            weight[iou > 0.5] -= iou[iou > 0.5]

        scores[idxs[1:]] *= weight
        idxs = idxs[1:][scores[idxs[1:]] > score_thresh]

    return torch.tensor(keep, device=boxes.device)


# ===================== 2. 自定义 Soft-NMS Predictor（兼容所有版本） =====================
class SoftNMSPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        # 适配不同版本的返回格式
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            preds = preds[0]

        # 执行原生后处理
        preds = super().postprocess(preds, img, orig_imgs)

        # 应用 Soft-NMS
        for i, det in enumerate(preds):
            if det is None or len(det) == 0:
                continue

            boxes = det.boxes.xyxy
            scores = det.boxes.conf

            keep_idx = soft_nms_pytorch(
                boxes=boxes,
                scores=scores.clone(),
                sigma=0.5,
                score_thresh=0.001,
                method="gaussian"
            )

            det.boxes.data = det.boxes.data[keep_idx]

        return preds


# ===================== 3. 批量推理MOT17 FRCNN序列（解决内存警告） =====================
def batch_infer_mot17_frcnn():
    # 配置参数
    MODEL_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_finetune_baseline\weights\best.pt"
    MOT17_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17\train"  # MOT17训练集根目录
    DEVICE = 0 if torch.cuda.is_available() else "cpu"

    # 待推理的MOT17 FRCNN序列列表
    frcnn_sequences = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN"
    ]

    # 加载模型（只需加载一次）
    model = YOLO(MODEL_PATH)
    print(f"✅ 模型加载完成：{MODEL_PATH}")
    print(f"✅ 推理设备：{DEVICE}")
    print(f"✅ 待推理序列数量：{len(frcnn_sequences)}")
    print("=" * 60)

    # 汇总统计结果
    total_stats = {
        "sequence": [],
        "total_images": [],
        "total_boxes": [],
        "avg_boxes_per_img": []
    }

    # 遍历每个序列推理
    for seq_name in frcnn_sequences:
        print(f"\n🔍 开始推理序列：{seq_name}")

        # 构建序列图片路径
        source_path = os.path.join(MOT17_ROOT, seq_name, "img1")
        if not os.path.exists(source_path):
            print(f"❌ 序列路径不存在：{source_path}，跳过")
            continue

        # 构建结果保存名称（区分不同序列）
        save_name = f"mot17_softnms_{seq_name}"

        # 🌟 核心修改：加入 stream=True 启用生成器模式，解决内存警告
        results_generator = model.predict(
            source=source_path,
            predictor=SoftNMSPredictor,
            conf=0.01,
            iou=0.7,
            max_det=300,
            device=DEVICE,
            save=True,
            save_txt=True,
            save_conf=True,
            name=save_name,
            show_labels=True,
            show_conf=False,
            line_width=1,
            verbose=False,  # 关闭冗余输出
            stream=True  # 启用生成器模式，逐张处理释放内存
        )

        # 遍历生成器统计结果（逐张处理，避免内存堆积）
        seq_total_images = 0
        seq_total_boxes = 0
        seq_box_stats = []
        save_dir = None  # 保存结果目录

        for i, det in enumerate(results_generator):
            seq_total_images += 1
            # 记录保存目录（仅第一次获取）
            if save_dir is None:
                save_dir = det.save_dir

            # 统计框数量
            if det is not None and len(det):
                box_num = len(det.boxes)
                seq_total_boxes += box_num
                seq_box_stats.append(box_num)
                # 打印前10张图的框数量（避免输出过多）
                if i < 10:
                    print(f"   第{i + 1}张图 - 保留框数量：{box_num}")

        # 计算该序列统计指标
        seq_avg_boxes = seq_total_boxes / seq_total_images if seq_total_images > 0 else 0

        # 输出该序列结果
        print(f"\n📊 序列 {seq_name} 推理完成：")
        print(f"   总图片数：{seq_total_images}")
        print(f"   总框数量：{seq_total_boxes}")
        print(f"   平均每图框数量：{seq_avg_boxes:.2f}")
        print(f"   结果保存目录：{save_dir}")
        print("-" * 50)

        # 加入汇总统计
        total_stats["sequence"].append(seq_name)
        total_stats["total_images"].append(seq_total_images)
        total_stats["total_boxes"].append(seq_total_boxes)
        total_stats["avg_boxes_per_img"].append(seq_avg_boxes)

    # 输出所有序列汇总结果
    print("\n" + "=" * 60)
    print("📈 所有MOT17 FRCNN序列推理汇总")
    print("=" * 60)
    print(f"{'序列名称':<15} {'总图片数':<10} {'总框数':<10} {'平均每图框数':<15}")
    print("-" * 60)
    for i in range(len(total_stats["sequence"])):
        seq = total_stats["sequence"][i]
        img_num = total_stats["total_images"][i]
        box_num = total_stats["total_boxes"][i]
        avg_box = total_stats["avg_boxes_per_img"][i]
        print(f"{seq:<15} {img_num:<10} {box_num:<10} {avg_box:<15.2f}")

    # 计算整体平均值
    if total_stats["total_images"]:
        overall_total_img = sum(total_stats["total_images"])
        overall_total_box = sum(total_stats["total_boxes"])
        overall_avg_box = overall_total_box / overall_total_img if overall_total_img > 0 else 0
        print("-" * 60)
        print(f"{'总计':<15} {overall_total_img:<10} {overall_total_box:<10} {overall_avg_box:<15.2f}")
    print("=" * 60)


# ===================== 主函数 =====================
if __name__ == "__main__":
    batch_infer_mot17_frcnn()