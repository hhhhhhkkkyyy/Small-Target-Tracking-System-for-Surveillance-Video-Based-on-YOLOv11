# ===================== imports =====================
import os
import cv2
import time
import torch
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict
import logging
import motmetrics as mm

from dadada import torchreid
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.application_util import preprocessing

# ===================== Config =====================
class Config:
    YOLO_WEIGHTS = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_final_continue\weights\best.pt"
    MOT17_TRAIN_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17\train"

    TARGET_SEQUENCES = [
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN"
    ]

    CONF_THRES = 0.25
    IOU_THRES = 0.45

    MAX_COSINE_DISTANCE = 0.3
    NN_BUDGET = 100
    MAX_AGE = 50
    MIN_HITS = 3

    DEVICE = "cuda:0"

    REID_WEIGHTS = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\reid_finetune_output\model.pth.tar-10"

    OUTPUT_DIR = "runs/track_eval_final"
    SAVE_VIDEO = False

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ===================== Soft-NMS =====================
def soft_nms(boxes, scores, sigma=0.5, thresh=0.001):
    if len(boxes) == 0:
        return []

    boxes = boxes.copy()
    scores = scores.copy()
    keep = []

    while scores.size > 0:
        idx = np.argmax(scores)
        keep.append(idx)

        if scores.size == 1:
            break

        box = boxes[idx]
        rest = np.delete(boxes, idx, axis=0)
        iou = compute_iou(box, rest)

        scores = np.delete(scores, idx)
        boxes = rest
        scores *= np.exp(-(iou ** 2) / sigma)

        mask = scores > thresh
        scores = scores[mask]
        boxes = boxes[mask]

    return keep

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)

# ===================== ReID Extractor =====================
class ReIDExtractor:
    def __init__(self):
        self.device = cfg.DEVICE
        self.model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1000,
            pretrained=False
        )

        state = torch.load(cfg.REID_WEIGHTS, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

        self.model.to(self.device).eval()

        from torchvision import transforms
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        print("✅ ReID 模型加载完成（不再训练）")

    @torch.no_grad()
    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(img).unsqueeze(0).to(self.device)
        feat = self.model(x).cpu().numpy().flatten()
        return feat / (np.linalg.norm(feat) + 1e-6)

# ===================== Tracker =====================
class MOTTracker:
    def __init__(self):
        self.detector = YOLO(cfg.YOLO_WEIGHTS).to(cfg.DEVICE)
        self.reid = ReIDExtractor()

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", cfg.MAX_COSINE_DISTANCE, cfg.NN_BUDGET
        )
        self.tracker = Tracker(metric, max_age=cfg.MAX_AGE)
        self.results = defaultdict(list)
        self.frame_id = 0

    def update(self, frame):
        detections = []

        results = self.detector(frame,verbose=False, conf=cfg.CONF_THRES, iou=cfg.IOU_THRES)[0]

        boxes, scores, feats = [], [], []

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            boxes.append([x1,y1,x2,y2])
            scores.append(float(box.conf[0]))
            feats.append(self.reid(crop))

        if boxes:
            keep = soft_nms(np.array(boxes), np.array(scores))
            for i in keep:
                x1,y1,x2,y2 = boxes[i]
                detections.append(
                    Detection([x1,y1,x2-x1,y2-y1], scores[i], feats[i])
                )

        self.tracker.predict()
        self.tracker.update(detections)

        tracks_out = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            x,y,w,h = t.to_tlwh()
            tracks_out.append([x,y,x+w,y+h,t.track_id])

        self.results[self.frame_id] = tracks_out
        self.frame_id += 1
        return tracks_out

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
    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(track_results.keys()) | set(gt_results.keys()))
    for f in frames:
        gt_boxes, gt_ids = [], []
        tr_boxes, tr_ids = [], []

        for g in gt_results.get(f, []):
            gt_boxes.append(g[:4])
            gt_ids.append(g[4])

        for t in track_results.get(f, []):
            tr_boxes.append(t[:4])
            tr_ids.append(t[4])

        if not gt_boxes and not tr_boxes:
            acc.update([], [], [])
            continue

        iou = np.zeros((len(gt_boxes), len(tr_boxes)))
        for i, g in enumerate(gt_boxes):
            for j, t in enumerate(tr_boxes):
                iou[i,j] = calculate_iou(g, t)

        acc.update(gt_ids, tr_ids, 1 - iou)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        "mota","idf1","num_switches","num_false_positives","num_misses"
    ])

    return {
        "MOTA": float(summary["mota"]),
        "IDF1": float(summary["idf1"]),
        "ID_switch": int(summary["num_switches"]),
        "FP": int(summary["num_false_positives"]),
        "FN": int(summary["num_misses"])
    }


def load_mot17_gt(gt_path):
    gt = defaultdict(list)
    with open(gt_path, 'r') as f:
        for line in f:
            fr, tid, x, y, w, h, conf, cls, vis = map(float, line.strip().split(','))
            if int(cls) == 1 and conf == 1 and vis > 0.2:
                gt[int(fr)-1].append([x, y, x+w, y+h, int(tid)])
    return gt
# ===================== Main =====================
def run():
    print("🚀 开始 MOT17 Tracking + Evaluation（无逐帧输出）")

    all_seq_metrics = []

    for seq in cfg.TARGET_SEQUENCES:
        print(f"\n{'='*70}")
        print(f"▶ Processing sequence: {seq}")
        print(f"{'='*70}")

        tracker = MOTTracker()

        img_dir = Path(cfg.MOT17_TRAIN_ROOT) / seq / "img1"
        gt_path = Path(cfg.MOT17_TRAIN_ROOT) / seq / "gt/gt.txt"

        frames = sorted(img_dir.glob("*.jpg"), key=lambda x: int(x.stem))
        gt_results = load_mot17_gt(gt_path)

        # ===================== Tracking Loop（无输出） =====================
        for p in tqdm(frames, desc=seq, leave=False):
            frame = cv2.imread(str(p))
            tracker.update(frame)

        # ===================== Evaluation（只做一次） =====================
        metrics = compute_clear_mot_metrics(tracker.results, gt_results)
        all_seq_metrics.append(metrics)

        # ===================== 序列级输出 =====================
        print(f"📊 [{seq}] Evaluation Results:")
        print(f"  MOTA        : {metrics['MOTA']:.4f}")
        print(f"  IDF1        : {metrics['IDF1']:.4f}")
        print(f"  ID Switches : {metrics['ID_switch']}")
        print(f"  FP / FN     : {metrics['FP']} / {metrics['FN']}")

    # ===================== 全局汇总 =====================
    print(f"\n{'='*70}")
    print("🏁 Overall MOT17 Train Results")
    print(f"{'='*70}")

    for key in all_seq_metrics[0].keys():
        values = [m[key] for m in all_seq_metrics]
        if isinstance(values[0], float):
            print(f"{key:<12}: {np.mean(values):.4f}")
        else:
            print(f"{key:<12}: {sum(values)}")



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
