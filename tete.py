import os
import cv2
import time
import torch
import numpy as np
import motmetrics as mm
import warnings
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict
import logging
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection as DS_Detection
from deep_sort.deep_sort.tracker import Tracker as DS_Tracker
from deep_sort.application_util import preprocessing

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection

# ===================== 配置类 =====================
class Config:
    YOLO_WEIGHTS = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_finetune_baseline\weights\best.pt"
    MOT17_TRAIN_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17\train"
    TARGET_SEQUENCES = ["MOT17-02-FRCNN","MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN",
                        "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
    MAX_COSINE_DISTANCE = 0.4
    # MAX_COSINE_DISTANCE = 0.4
    # MAX_AGE = 30
    NN_BUDGET = 100
    MAX_AGE = 30
    MIN_HITS = 3
    CONF_THRES = 0.25
    IOU_THRES = 0.45
    REID_WEIGHTS = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\reid\osnet_x0_25_imagenet.pth"
    DEVICE = "cuda:0"
    OUTPUT_DIR = "runs/track_eval_mot17_final_2"
    SAVE_VIDEO = True
    SAVE_METRICS = True
    # Soft-NMS参数补充
    SOFT_NMS_SIGMA = 0.5
    SOFT_NMS_THRESH = 0.001


# 初始化配置
cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ===================== 1. Soft-NMS核心实现 =====================
def soft_nms_pytorch(boxes, scores, sigma=cfg.SOFT_NMS_SIGMA, score_thresh=cfg.SOFT_NMS_THRESH, method="gaussian"):
    if len(boxes) == 0:
        return []

    boxes_tlbr = np.copy(boxes)
    boxes_tlbr[:, 2] = boxes_tlbr[:, 0] + boxes_tlbr[:, 2]
    boxes_tlbr[:, 3] = boxes_tlbr[:, 1] + boxes_tlbr[:, 3]

    keep = []
    scores_copy = scores.copy()
    idxs = np.argsort(scores_copy)[::-1]

    while len(idxs) > 0:
        current_idx = idxs[0]
        keep.append(current_idx)

        if len(idxs) == 1:
            break
        rest_idxs = idxs[1:]
        iou = _calc_iou(boxes_tlbr[current_idx], boxes_tlbr[rest_idxs])

        if method == "gaussian":
            weight = np.exp(-(iou ** 2) / sigma)
        else:
            weight = 1 - iou
            weight[iou > 0.5] = 0

        scores_copy[rest_idxs] *= weight
        idxs = idxs[1:][scores_copy[rest_idxs] > score_thresh]
        idxs = idxs[np.argsort(scores_copy[idxs])[::-1]]

    return keep


def _calc_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter

    return inter / (union + 1e-6)


# ===================== 2. ReID特征提取器（OSNet-x0.25 适配小目标） =====================
class ReIDExtractor:
    def __init__(self, reid_model_weights):
        """
        适配GAM轻量化特征融合设计：
        1. OSNet-x0.25轻量化结构，与GAM的池化+卷积+全连接层轻量化设计匹配
        2. 特征提取后L2归一化，强化小目标微弱特征的区分度
        """
        self.device = cfg.DEVICE
        import torchreid

        # 构建轻量化OSNet-x0.25，适配小目标特征提取
        self.model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1000,
            loss='softmax',
            pretrained=False
        )

        # 加载权重
        state_dict = torch.load(reid_model_weights, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # OSNet官方推荐输入尺寸，适配小目标裁剪区域
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"✅ 使用ReID模型：OSNet-x0.25（适配小目标轻量化特征提取）")
        print(f"✅ 加载ReID权重：{reid_model_weights}")

    @torch.no_grad()
    def extract_feature(self, image_crop):
        # 优化1：空裁剪区域鲁棒处理
        if image_crop is None or image_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_crop).unsqueeze(0).to(self.device)

        # 提取512维特征，与GAM输出的增强特征维度匹配
        feat = self.model(tensor)
        feat = feat.cpu().numpy().reshape(-1)

        # L2归一化，强化小目标特征区分度（适配GAM的特征融合逻辑）
        feat /= (np.linalg.norm(feat) + 1e-6)
        return feat


# ===================== 3. DeepSORT跟踪器（优化过滤逻辑+适配GAM特征） =====================
class YOLOv8DeepSORTTracker:
    def __init__(self):
        self.yolo_model = YOLO(cfg.YOLO_WEIGHTS).to(cfg.DEVICE)
        # YOLO加载的模型已集成GAM注意力机制：
        # - 通道注意力筛选小目标关键特征通道
        # - 空间注意力聚焦小目标空间位置
        # - 特征融合层实现通道权重×空间权重×原始特征，强化小目标表达
        print(f"✅ 加载YOLO权重（集成GAM轻量化注意力）：{cfg.YOLO_WEIGHTS}")

        self.reid_extractor = ReIDExtractor(cfg.REID_WEIGHTS)

        # 初始化跟踪器，匹配GAM增强后的特征维度
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", cfg.MAX_COSINE_DISTANCE, cfg.NN_BUDGET
        )
        self.tracker = DS_Tracker(metric, max_age=cfg.MAX_AGE)

        self.track_history = defaultdict(list)
        self.frame_idx = 0
        self.track_results = {}
        self.video_writers = {}
        self.min_hits = cfg.MIN_HITS

    def init_video_writer(self, seq_name, frame_shape):
        if seq_name not in self.video_writers:
            video_dir = Path(cfg.OUTPUT_DIR) / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{seq_name}_deep_sort_track.mp4"

            height, width = frame_shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writers[seq_name] = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
        return self.video_writers[seq_name]

    def preprocess_detections(self, frame):
        results = self.yolo_model(frame, conf=cfg.CONF_THRES, iou=cfg.IOU_THRES)[0]
        detections = []

        # 提取YOLO检测结果（GAM增强后的小目标特征）
        boxes = []
        scores = []
        features = []
        for box in results.boxes:
            if int(box.cls[0].cpu().numpy()) != 0:  # 只保留行人
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy

            # 优化2：裁剪区域边界检查，避免小目标越界导致空特征
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))

            # 跳过无效裁剪（小目标完全越界）
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            tlwh = [x1, y1, x2 - x1, y2 - y1]
            conf = box.conf[0].cpu().numpy()

            # 提取小目标ReID特征（适配GAM增强后的特征）
            crop = frame[y1:y2, x1:x2]
            feature = self.reid_extractor.extract_feature(crop)

            boxes.append(tlwh)
            scores.append(conf)
            features.append(feature)

        # 应用Soft-NMS，保留GAM增强后的小目标检测框
        if len(boxes) > 0:
            boxes_np = np.array(boxes)
            scores_np = np.array(scores)
            keep_indices = soft_nms_pytorch(boxes_np, scores_np)

            # 过滤检测结果
            boxes = [boxes[i] for i in keep_indices]
            scores = [scores[i] for i in keep_indices]
            features = [features[i] for i in keep_indices]

        # 转换为DeepSORT的Detection格式
        for box, score, feature in zip(boxes, scores, features):
            detections.append(DS_Detection(box, score, feature))

        return detections

    def update_tracker(self, frame):
        detections = self.preprocess_detections(frame)
        self.tracker.predict()
        self.tracker.update(detections)

        current_tracks = []
        for track in self.tracker.tracks:
            # 优化3：调整过滤逻辑，避免重复过滤小目标
            # track.is_confirmed()已包含hits >= n_init（默认3），仅补充time_since_update过滤
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            tlwh = track.to_tlwh()
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            current_tracks.append([x1, y1, x2, y2, track.track_id])

            # 更新跟踪历史（适配小目标轨迹可视化）
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.track_history[track.track_id].append((center_x, center_y))
            if len(self.track_history[track.track_id]) > 50:
                self.track_history[track.track_id].pop(0)

        self.track_results[self.frame_idx] = current_tracks
        self.frame_idx += 1
        return current_tracks

    def draw_visualization(self, frame, tracks, gt_boxes=None):
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if gt_boxes is not None:
            for gt in gt_boxes:
                x1, y1, x2, y2, gt_id = map(int, gt[:5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, f"GT:{gt_id}", (x1, max(y1 - 20, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.putText(frame, f"Frame: {self.frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame


# 禁用ultralytics日志输出
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
logging.getLogger('ultralytics.engine.predictor').setLevel(logging.CRITICAL)
logging.getLogger('ultralytics.nn.tasks').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')


# ===================== 4. 工具函数 =====================
def get_mot17_train_sequences():
    sequences = []
    for seq_name in cfg.TARGET_SEQUENCES:
        seq_path = Path(cfg.MOT17_TRAIN_ROOT) / seq_name
        if not seq_path.exists():
            print(f"⚠️ 跳过不存在的序列：{seq_path}")
            continue

        img_dir = seq_path / "img1"
        frame_paths = sorted(img_dir.glob("*.jpg"), key=lambda x: int(x.stem))
        if len(frame_paths) == 0:
            print(f"⚠️ 序列{seq_name}无图片")
            continue

        gt_path = seq_path / "gt" / "gt.txt"
        sequences.append({
            "name": seq_name,
            "frame_paths": frame_paths,
            "gt_path": str(gt_path),
            "img_shape": cv2.imread(str(frame_paths[0])).shape[:2]
        })
    return sequences


def load_mot17_gt(gt_path):
    gt_results = {}
    if not os.path.exists(gt_path):
        return gt_results

    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                frame_id = int(float(parts[0])) - 1
                gt_id = int(float(parts[1]))
                x1 = float(parts[2])
                y1 = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])
                cls = int(float(parts[7])) if len(parts) >= 8 else 1
                visibility = float(parts[8]) if len(parts) >= 9 else 1.0

                if cls == 1 and conf == 1.0 and visibility > 0.2 and w > 0 and h > 0:
                    x2 = x1 + w
                    y2 = y1 + h
                    if frame_id not in gt_results:
                        gt_results[frame_id] = []
                    gt_results[frame_id].append((x1, y1, x2, y2, gt_id))
            except:
                continue
    return gt_results


def calculate_iou(box1, box2):
    """手动计算IOU（兼容所有motmetrics版本）"""
    # box1/box2: [x1,y1,x2,y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if box1_area + box2_area - inter_area == 0:
        return 0.0
    return inter_area / (box1_area + box2_area - inter_area)


def compute_clear_mot_metrics(track_results, gt_results):
    """
    计算 CLEAR MOT 指标：
    MOTA / IDF1 / ID Switch / MT / ML
    """
    accumulator = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(track_results.keys()) | set(gt_results.keys()))

    # 记录 GT 生命周期
    gt_lifetime = defaultdict(lambda: {"total": 0, "tracked": 0})

    for frame_id in all_frames:
        gt_boxes, gt_ids = [], []
        if frame_id in gt_results:
            for gt in gt_results[frame_id]:
                gt_boxes.append(gt[:4])
                gt_ids.append(gt[4])
                gt_lifetime[gt[4]]["total"] += 1

        trk_boxes, trk_ids = [], []
        if frame_id in track_results:
            for trk in track_results[frame_id]:
                trk_boxes.append(trk[:4])
                trk_ids.append(trk[4])

        if len(gt_boxes) == 0 and len(trk_boxes) == 0:
            accumulator.update([], [], [])
            continue

        iou_matrix = np.zeros((len(gt_boxes), len(trk_boxes)), dtype=np.float32)
        for i, gt in enumerate(gt_boxes):
            for j, trk in enumerate(trk_boxes):
                iou_matrix[i, j] = calculate_iou(gt, trk)

        accumulator.update(
            gt_ids,
            trk_ids,
            1 - iou_matrix
        )

        # 统计 GT 被成功匹配的帧数（IoU > 0.5）
        if len(gt_boxes) > 0 and len(trk_boxes) > 0:
            row, col = mm.lap.linear_sum_assignment(1 - iou_matrix)
            for r, c in zip(row, col):
                if iou_matrix[r, c] > 0.5:
                    gt_lifetime[gt_ids[r]]["tracked"] += 1

    # ===== motmetrics 官方指标 =====
    mh = mm.metrics.create()
    summary = mh.compute(
        accumulator,
        metrics=[
            "mota",
            "idf1",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_objects"
        ],
        name="CLEAR_MOT"
    )

    # ===== MT / ML =====
    mt, ml = 0, 0
    for gt_id, stat in gt_lifetime.items():
        if stat["total"] == 0:
            continue
        ratio = stat["tracked"] / stat["total"]
        if ratio >= 0.8:
            mt += 1
        elif ratio <= 0.2:
            ml += 1

    total_gt = len(gt_lifetime)

    return {
        "MOTA": round(float(summary["mota"]), 4),
        "IDF1": round(float(summary["idf1"]), 4),
        "ID_switches": int(summary["num_switches"]),
        "MT": mt,
        "ML": ml,
        "MT_ratio": round(mt / total_gt, 4) if total_gt > 0 else 0.0,
        "ML_ratio": round(ml / total_gt, 4) if total_gt > 0 else 0.0,
        "False_Positives": int(summary["num_false_positives"]),
        "Misses": int(summary["num_misses"]),
        "Total_GT": int(summary["num_objects"])
    }


# ===================== 主函数 =====================
def run_mot17_evaluation():
    train_sequences = get_mot17_train_sequences()
    if not train_sequences:
        raise FileNotFoundError("未找到有效MOT17序列，请检查路径！")
    print(f"\n✅ 找到 {len(train_sequences)} 个序列：{[s['name'] for s in train_sequences]}")

    total_metrics = defaultdict(list)
    total_fps = []
    metrics_file = Path(cfg.OUTPUT_DIR) / "evaluation_metrics_softnms.txt"

    for seq_idx, seq in enumerate(train_sequences):
        seq_name = seq['name']
        frame_paths = seq['frame_paths']
        gt_path = seq['gt_path']
        img_shape = seq['img_shape']

        print(f"\n{'=' * 70}")
        print(f"处理序列 {seq_idx + 1}/{len(train_sequences)}：{seq_name}（{len(frame_paths)}帧）")
        print(f"{'=' * 70}")

        gt_results = load_mot17_gt(gt_path)
        has_gt = len(gt_results) > 0
        if not has_gt:
            print(f"⚠️ 序列{seq_name}无有效GT标注，跳过指标计算")

        tracker = YOLOv8DeepSORTTracker()
        seq_fps = []

        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=f"{seq_name} 处理中")):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"⚠️ 跳过损坏帧：{frame_path}")
                continue

            start_time = time.time()
            tracks = tracker.update_tracker(frame)

            current_gt = gt_results.get(frame_idx, [])
            frame_vis = tracker.draw_visualization(frame, tracks, current_gt)

            if cfg.SAVE_VIDEO:
                try:
                    video_writer = tracker.init_video_writer(seq_name, img_shape)
                    video_writer.write(frame_vis)
                except Exception as e:
                    print(f"⚠️ 视频写入失败：{e}")

            fps = 1.0 / (time.time() - start_time + 1e-6)
            seq_fps.append(fps)
            total_fps.append(fps)

        # 释放视频写入器
        if seq_name in tracker.video_writers:
            tracker.video_writers[seq_name].release()

        seq_result = {
            "name": seq_name,
            "avg_fps": np.mean(seq_fps) if seq_fps else 0,
            "metrics": None,
            "has_gt": has_gt
        }

        if has_gt:
            seq_metrics = compute_clear_mot_metrics(tracker.track_results, gt_results)
            seq_result["metrics"] = seq_metrics

        # 收集总指标
        if seq_result["metrics"]:
            total_metrics['MOTA'].append(seq_result['metrics']['MOTA'])
            total_metrics['IDF1'].append(seq_result['metrics']['IDF1'])
        else:
            total_metrics['MOTA'].append(0.0)
            total_metrics['IDF1'].append(0.0)

        # 打印序列结果
        print(f"\n📊 【{seq_name}】指标：")
        if seq_result["metrics"]:
            print(f"  MOTA = {seq_result['metrics']['MOTA']:.4f} | IDF1 = {seq_result['metrics']['IDF1']:.4f}")
            print(
                f"  MT = {seq_result['metrics']['MT']} ({seq_result['metrics']['MT_ratio']:.4f}) | ML = {seq_result['metrics']['ML']} ({seq_result['metrics']['ML_ratio']:.4f})")
            print(
                f"  ID切换 = {seq_result['metrics']['ID_switches']} | 误检 = {seq_result['metrics']['False_Positives']} | 漏检 = {seq_result['metrics']['Misses']}")
        else:
            print("  无有效GT标注，无法计算MOTA/IDF1")
        print(f"  平均FPS = {seq_result['avg_fps']:.2f}")

    # 保存评估报告
    if cfg.SAVE_METRICS:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MOT17 Train集评估报告（Soft-NMS + DeepSORT + ReID + GAM注意力）\n")
            f.write("=" * 60 + "\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试序列: {cfg.TARGET_SEQUENCES}\n")
            f.write(f"YOLO权重: {cfg.YOLO_WEIGHTS}\n")
            f.write(f"ReID权重: {cfg.REID_WEIGHTS}\n")
            f.write(f"Soft-NMS参数: sigma={cfg.SOFT_NMS_SIGMA}, thresh={cfg.SOFT_NMS_THRESH}\n")
            f.write(f"跟踪器参数: max_age={cfg.MAX_AGE}, min_hits={cfg.MIN_HITS}, NN_BUDGET={cfg.NN_BUDGET}\n")
            f.write("\n【GAM注意力机制适配说明】\n")
            f.write("1. 通道注意力筛选小目标关键特征通道，空间注意力聚焦小目标空间位置\n")
            f.write("2. 特征融合层实现通道权重×空间权重×原始特征，强化小目标微弱特征表达\n")
            f.write("3. 轻量化设计，与OSNet-x0.25协同保证推理速度\n")
            f.write("\n【全局指标】\n")
            f.write(f"平均MOTA: {np.mean(total_metrics['MOTA']):.4f}\n")
            f.write(f"平均IDF1: {np.mean(total_metrics['IDF1']):.4f}\n")
            f.write(f"平均FPS: {np.mean(total_fps):.2f}\n")
            f.write("\n【各序列详细指标】\n")
            for i, seq in enumerate(train_sequences):
                f.write(f"\n{seq['name']}：\n")
                if total_metrics['MOTA'][i] > 0:
                    f.write(f"  MOTA: {total_metrics['MOTA'][i]:.4f}\n")
                    f.write(f"  IDF1: {total_metrics['IDF1'][i]:.4f}\n")
                else:
                    f.write(f"  MOTA: N/A\n")
                    f.write(f"  IDF1: N/A\n")
                f.write(
                    f"  平均FPS: {np.mean([fps for j, fps in enumerate(total_fps) if j < len(train_sequences[i]['frame_paths'])]) if train_sequences[i]['frame_paths'] else 0:.2f}\n")
            print(f"\n✅ 完整评估报告已保存至：{metrics_file}")

    print(f"\n🎉 评估完成！")
    print(f"📈 全局平均MOTA: {np.mean(total_metrics['MOTA']):.4f}")
    print(f"📈 全局平均IDF1: {np.mean(total_metrics['IDF1']):.4f}")
    print(f"⚡ 全局平均FPS: {np.mean(total_fps):.2f}")


if __name__ == "__main__":
    try:
        assert os.path.exists(cfg.YOLO_WEIGHTS), f"YOLO权重不存在：{cfg.YOLO_WEIGHTS}"
        assert os.path.exists(cfg.MOT17_TRAIN_ROOT), f"MOT17 Train路径不存在：{cfg.MOT17_TRAIN_ROOT}"
        assert os.path.exists(cfg.REID_WEIGHTS), f"ReID权重不存在：{cfg.REID_WEIGHTS}"

        run_mot17_evaluation()  # Start evaluation
    except Exception as e:
        print(f"\n❌ 运行失败：{str(e)}")
        import traceback

        traceback.print_exc()