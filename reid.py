import cv2
import torch
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision import transforms
from bytetrack.tracker import BYTETracker
from reid.extractor import ReIDExtractor
from reid.osnet import models
import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. 路径配置
# =========================
SEQ_NAMES = [
    "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
    "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"
]  # 所有的 MOT17 序列

# 输入图像路径
MOT_ROOT = Path(r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17\train")

# 检测结果路径（已经有的 det.txt 文件）
DET_ROOT = Path(r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\mot17_bytetrack_det")

# 输出结果路径
SAVE_ROOT = Path("runs/bytetrack_reid")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# ReID 模型路径
REID_CKPT = Path(r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\reid\osnet_x0_25_imagenet.pth")

# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. ReID 特征提取器
# =========================
class OSNet(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, feat_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class ReIDExtractor:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = OSNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 128)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def __call__(self, img, boxes):
        feats = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                feats.append(np.zeros(512))
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = self.tf(crop).unsqueeze(0).to(self.device)
            feat = self.model(crop).cpu().numpy()[0]
            feats.append(feat)
        return np.array(feats)


# =========================
# 3. 加载检测结果（det.txt）
# =========================
def load_dets(det_file):
    dets = {}
    with open(det_file, "r") as f:
        for line in f:
            items = line.strip().split(",")
            frame = int(items[0])
            x, y, w, h, s = map(float, items[2:7])
            dets.setdefault(frame, []).append([x, y, x + w, y + h, s])
    return dets


# =========================
# 4. 追踪与检测逻辑
# =========================
def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def associate(tracks, detections, feats_t, feats_d, iou_thres=0.3):
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.zeros((len(tracks), len(detections)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            cost[i, j] = 0.5 * (1 - iou(t.tlbr, d)) + 0.5 * np.dot(feats_t[i], feats_d[j])

    row, col = linear_sum_assignment(cost)

    matches, u_t, u_d = [], [], []
    for r, c in zip(row, col):
        if cost[r, c] < iou_thres:
            matches.append((r, c))
        else:
            u_t.append(r)
            u_d.append(c)

    return matches, u_t, u_d


# =========================
# 5. 主程序逻辑（批量处理所有序列）
# =========================
for SEQ_NAME in SEQ_NAMES:
    print(f"Processing sequence: {SEQ_NAME}")

    # 输入图像路径
    IMG_DIR = Path(r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17\train") / SEQ_NAME / "img1"
    # 检测结果路径（已经有的 det.txt 文件）
    DET_FILE = Path(r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\mot17_bytetrack_det") / SEQ_NAME / "det" / "det.txt"
    # 输出结果路径
    SAVE_DIR = Path(f"runs/bytetrack_reid/{SEQ_NAME}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_TXT = SAVE_DIR / f"{SEQ_NAME}.txt"

    # 创建 Tracker 和 ReID 提取器
    reid_extractor = ReIDExtractor(
        model_path=REID_CKPT,
        device=DEVICE
    )

    tracker = BYTETracker(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        use_reid=True
    )

    detections = load_dets(DET_FILE)
    results = []

    img_files = sorted(IMG_DIR.glob("*.jpg"), key=lambda x: int(x.stem))

    for img_path in img_files:
        frame_id = int(img_path.stem)
        img = cv2.imread(str(img_path))

        dets = np.array(detections.get(frame_id, []), dtype=np.float32)

        if len(dets) > 0:
            boxes = dets[:, :4]
            scores = dets[:, 4]

            # ReID 特征
            features = reid_extractor(img, boxes)

            online_targets = tracker.update(
                boxes, scores, features
            )
        else:
            online_targets = tracker.update(
                np.empty((0, 4)), np.empty((0,)), None
            )

        for t in online_targets:
            x1, y1, x2, y2 = t.tlbr
            tid = t.track_id

            results.append([
                frame_id,
                tid,
                x1, y1,
                x2 - x1,
                y2 - y1,
                1, -1, -1, -1
            ])

    # =========================
    # 6. 保存结果到文本文件
    # =========================
    with open(SAVE_TXT, "w") as f:
        for r in results:
            f.write(",".join(map(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x), r)) + "\n")

    print(f"✅ Tracking finished for {SEQ_NAME}")
    print(f"📁 Result saved to: {SAVE_TXT}")
