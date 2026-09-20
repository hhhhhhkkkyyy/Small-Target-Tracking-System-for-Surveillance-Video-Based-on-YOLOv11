# 基于 YOLO11 的监控视频小目标跟踪系统设计与实现

一个面向监控视频场景的**小目标多目标跟踪（Multi-Object Tracking, MOT）系统**。系统以 **YOLO11** 作为检测器，针对密集行人与小目标场景引入了 **GAM 全局注意力机制、小目标特征增强分支与 SIoU 损失**，并使用 **DeepSORT / ByteTrack + OSNet 行人重识别（ReID）** 完成跨帧目标关联，最后通过 **Soft-NMS** 缓解密集场景下的漏检与误检。项目同时提供了一个基于 **Tkinter** 的桌面 GUI，支持视频 / 摄像头输入、实时可视化、ID 轨迹绘制与结果视频导出。

---

## 目录

- [项目简介](#项目简介)
- [系统流程](#系统流程)
- [功能特性](#功能特性)
- [环境依赖](#环境依赖)
- [目录结构](#目录结构)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [模型训练](#模型训练)
- [跟踪与评估](#跟踪与评估)
- [GUI 使用说明](#gui-使用说明)
- [实验结果](#实验结果)
- [改进点说明](#改进点说明)
- [常见问题](#常见问题)
- [参考与致谢](#参考与致谢)
- [许可证](#许可证)

---

## 项目简介

监控视频中的小目标（远处行人、密集人群中的个体）普遍存在分辨率低、纹理弱、易被遮挡等特点，给检测与跟踪带来较大挑战。本系统在 YOLO11 检测框架的基础上进行针对性改进，并结合 DeepSORT / ByteTrack 跟踪器与 OSNet ReID 特征，构建了一套完整的**检测 → 跟踪 → 评估 → 可视化**流水线。

系统的主要组成：

1. **检测器**：YOLO11（`ultralytics`），支持 `yolo11n/s/m` 多个规格的预训练权重。
2. **改进模块**：GAM 全局注意力、小目标特征增强分支、SIoU 边界框回归损失。
3. **跟踪器**：DeepSORT（`deep_sort/`）与 ByteTrack（`bytetrack/`），可切换。
4. **重识别（ReID）**：OSNet-x0.25 / OSNet-x1.0，用于低置信度目标的外观关联。
5. **后处理**：Soft-NMS 抑制密集重叠框，减少漏检。
6. **评估**：基于 `motmetrics` 计算 MOTA、IDF1、ID Switch、FP、FN 等指标。
7. **桌面 GUI**：`gggggg.py`，一键完成视频 / 摄像头实时跟踪与结果导出。

---

## 系统流程

```text
输入视频 / 摄像头
        │
        ▼
┌─────────────────┐
│  YOLO11 检测器   │  (可选：GAM + 小目标增强 + SIoU)
└────────┬────────┘
         │ 检测框 + 置信度
         ▼
┌─────────────────┐
│     Soft-NMS     │  缓解密集重叠 / 遮挡
└────────┬────────┘
         │ 筛选后的检测框
         ▼
┌─────────────────────────────┐
│  外观特征提取 (OSNet ReID)    │  低置信度目标计算外观特征
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  数据关联 (DeepSORT / ByteTrack)│  卡尔曼滤波 + 匈牙利匹配
└────────┬────────────────────┘
         │ 跟踪轨迹 (ID + bbox)
         ▼
┌─────────────────────────────┐
│  可视化 + 指标评估 + 结果导出  │
└─────────────────────────────┘
```

---

## 功能特性

- **检测改进**：在 YOLO11 中嵌入 GAM（通道 + 空间注意力）与小目标特征增强分支，并将边界框损失替换为 SIoU，提升小目标与密集场景的检测能力。
- **双跟踪器**：同时集成 DeepSORT 与 ByteTrack，可在不同实验间切换对比。
- **外观关联**：使用 OSNet-x0.25 轻量 ReID 网络提取 512 维外观特征，改善遮挡与 ID 切换问题。
- **Soft-NMS**：用高斯加权 Soft-NMS 替代传统 NMS，保留密集小目标。
- **实时 / 高精度双模式**：高精度模式对低置信度目标逐帧计算 ReID；实时模式通过隔帧采样与数量上限控制计算量。
- **完整评估**：基于 `motmetrics` 输出 MOTA、IDF1、ID Switch、FP、FN，并支持保存带 ID 的结果视频。
- **桌面 GUI**：Tkinter 界面，支持视频文件 / 摄像头输入、置信度阈值调节、模型切换、结果视频导出。

---

## 环境依赖

推荐使用 Python 3.10，并配置支持 CUDA 的 PyTorch（无 GPU 时自动回退到 CPU）。

| 依赖 | 说明 |
| --- | --- |
| Python | 3.8+，推荐 3.10 |
| PyTorch | 2.x（按需安装 CUDA 版本） |
| ultralytics | 8.x，提供 YOLO11 支持 |
| torchvision | 与 PyTorch 匹配 |
| opencv-python | 视频读写与图像处理 |
| numpy / Pillow | 数值计算与图像显示 |
| scipy | 匈牙利匹配等 |
| motmetrics | MOT 指标评估 |
| tqdm | 进度显示 |

> `torchreid`（OSNet）已随项目内置在 `torchreid/` 目录中，无需额外安装。

安装示例（建议在虚拟环境中执行）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python pillow numpy scipy motmetrics tqdm
```

---

## 目录结构

```text
dadada/
├── gggggg.py                  # 主 GUI 入口（YOLO11 + DeepSORT + OSNet ReID）
├── gui.py                     # 早期精简版 GUI（YOLO11 内置 ByteTrack）
├── base_tr.py                 # CrowdHuman 预训练（基线 / 改进模型）
├── qianyi.py                  # MOT17 迁移 / 微调训练（含标注修正）
├── qianyi_by.py               # ByteTrack 方案迁移训练
├── qianyi_cr_by.py            # CrowdHuman + ByteTrack 迁移训练
├── qianyisoftnms.py           # 集成 Soft-NMS 的预测器
├── bebebe4.py                 # SIoU 损失 + GAM 小目标融合模块定义
├── test.py                    # 跟踪 + 评估（Soft-NMS + DeepSORT + ReID）
├── te1.py                     # 跟踪 + 评估（迁移权重版本）
├── tete.py                    # 跟踪 + 评估（GAM 适配版本）
├── reid.py                    # ByteTrack + ReID 跟踪脚本
├── dataset.py                 # MOT17 原始标注 → YOLO 格式转换
├── datasese.py                # MOT17 GT 整理为 submit 格式
├── bytetrack.yaml             # ByteTrack 跟踪器参数
├── mot17.yaml                 # MOT17 训练数据配置
├── crowdhuman/data.yaml       # CrowdHuman 数据配置
├── yolo11n.pt / yolo11s.pt / yolo11m.pt   # YOLO11 预训练权重
├── deep_sort/                 # DeepSORT 实现
├── bytetrack/                 # ByteTrack 实现
├── torchreid/                 # torchreid 库（OSNet 等模型）
├── reid/                      # ReID 权重（osnet_x0_25_imagenet.pth 等）
├── MOT17/                     # MOT17 原始数据集
├── MOT17_yolo/                # 转换后的 MOT17 YOLO 格式数据集
├── crowdhuman/                # CrowdHuman 数据集
├── mot17_bytetrack_det/       # ByteTrack 检测结果
├── mot17_softnms/             # Soft-NMS 检测结果
├── runs/                      # 训练权重与跟踪评估输出
└── output_*.mp4               # GUI 导出的结果视频
```

---

## 数据集准备

### 1. MOT17

从 [MOT Challenge](https://motchallenge.net/data/MOT17/) 下载 MOT17 数据集，解压后目录结构应包含 `MOT17/train` 与 `MOT17/test`。

项目默认只使用 **FRCNN 检测器**的序列（如 `MOT17-02-FRCNN`）。运行 `dataset.py` 将 MOT 格式的 `gt.txt` 转换为 YOLO 格式：

```bash
python dataset.py
```

转换结果保存在 `MOT17_yolo/`，并自动生成 `MOT17_yolo/data.yaml`。

### 2. CrowdHuman

CrowdHuman 用于密集行人预训练。可使用官方数据集，或使用仓库 `README.roboflow.txt` 中记录的 Roboflow 导出版本（10320 张图片，YOLOv11 格式）：

- 数据来源：<https://universe.roboflow.com/keio-dba-team/crowdhuman-nur7g>
- 许可：CC BY 4.0

将数据放到 `crowdhuman/` 下，并确保 `crowdhuman/data.yaml` 中的路径正确。

---

## 快速开始

1. 安装依赖（见 [环境依赖](#环境依赖)）。
2. 确认预训练权重 `yolo11s.pt` 以及 ReID 权重 `reid/osnet_x0_25_imagenet.pth` 存在。
3. 启动 GUI：

```bash
python gggggg.py
```

在 GUI 中选择视频或启用摄像头，点击「开始跟踪」即可。结果视频默认导出为当前目录下的 `output_<时间戳>.mp4`。

---

## 模型训练

### 基线模型（YOLO11s 从头训练）

可直接使用 `ultralytics` 训练，也可参考 `mot17_yolov11s_from_scratch` 实验配置：

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data="mot17.yaml",
    epochs=50,
    imgsz=1280,
    batch=4,
    name="mot17_baseline",
)
```

### 改进模型（GAM + 小目标增强 + SIoU）

采用「CrowdHuman 预训练 → MOT17 微调」两阶段策略：

1. 在 CrowdHuman 上预训练：`python base_tr.py`
2. 在 MOT17 上迁移微调：`python qianyi.py`

对应的权重目录包括 `crowdhuman_yolov11s_gam_siou1`、`mot17_1_baseline` 等，最终检测权重保存在 `runs/detect/<实验名>/weights/best.pt`。

> 训练脚本中的数据集路径为绝对路径，迁移到其他机器时请根据实际目录修改 `*.yaml` 与脚本中的路径。

---

## 跟踪与评估

跟踪与评估脚本（`test.py` / `te1.py` / `tete.py`）会：

1. 加载 YOLO 检测权重与 OSNet ReID 权重；
2. 对 MOT17 训练集序列执行检测 + Soft-NMS + ReID + DeepSORT；
3. 使用 `motmetrics` 计算 MOTA / IDF1 / ID Switch / FP / FN；
4. 输出指标报告与带 ID 的结果视频到 `runs/`。

运行示例：

```bash
python te1.py
```

脚本内的 `Config` 类可调整检测权重、ReID 权重、置信度阈值、Soft-NMS 参数以及要评估的序列列表。

---

## GUI 使用说明

运行 `python gggggg.py` 后，界面包含：

- **模型选择**：在「YOLO11s Baseline（从头训练）」与「YOLO11s + GAM + 小目标增强 + SIoU（改进模型）」之间切换。
- **检测置信度阈值**：0.1 ~ 0.9 可调。
- **运行模式**：高精度模式（低置信度目标逐帧计算 ReID）/ 实时模式（隔帧 + 数量限制）。
- **输入源**：选择本地视频文件，或勾选「使用摄像头实时跟踪」。
- **导出结果视频**：保存带 ID 与轨迹的 `output_*.mp4`。

GUI 内模型与 ReID 权重路径为硬编码，部署到新环境时请同步修改 `gggggg.py` 顶部的 `MODEL_PATHS` 与 `init_reid_model()` 中的权重路径。

---

## 实验结果

以下结果在 MOT17 训练集（FRCNN 序列）上评估，指标由 `motmetrics` 计算：

| 方案 | 检测权重 | 跟踪器 | MOTA | IDF1 | 平均 FPS |
| --- | --- | --- | --- | --- | --- |
| 基线 | mot17_finetune_baseline | DeepSORT + ReID + Soft-NMS | 0.8305 | 0.7475 | 4.78 |
| + GAM 注意力 | mot17_finetune_baseline | DeepSORT + ReID + Soft-NMS | 0.8315 | 0.7767 | 4.99 |
| 改进检测器（GAM + SIoU） | mot17_yolov11s_gam_siou | DeepSORT + ReID + Soft-NMS | 0.6886 | 0.6700 | 7.44 |
| ByteTrack 方案 | mot17_yolov11s_by | ByteTrack + ReID | 0.8106 | 0.7562 | 5.66 |
| 最终续训 | mot17_final_continue | DeepSORT + ReID + Soft-NMS | 0.8100 | 0.7577 | 4.62 |

各次实验的详细分序列指标见 `runs/track_eval_*/evaluation_metrics_softnms.txt`。

---

## 改进点说明

1. **GAM 全局注意力 + 小目标增强分支**
   通过通道注意力筛选小目标关键特征通道，空间注意力聚焦小目标位置，并以残差方式融合小目标增强分支，强化微弱特征表达。

2. **SIoU 损失**
   在 IoU 基础上引入角度、距离与形状约束，改善边界框回归精度，尤其适用于小目标。

3. **Soft-NMS**
   用高斯加权 Soft-NMS 替代硬阈值 NMS，降低密集 / 遮挡场景中的误删概率，提升召回。

4. **OSNet 轻量 ReID + DeepSORT**
   对低置信度目标提取 512 维外观特征，参与级联匹配，减少遮挡后的 ID 切换。

5. **实时 / 高精度双模式**
   通过置信度门控、隔帧采样与数量上限，在跟踪精度与推理速度之间取得平衡。

---

## 常见问题

**Q1：运行时报找不到 `torchreid` 模块？**

`torchreid` 已内置在项目根目录，请确保在项目根目录（`dadada/`）下运行脚本，或将该目录加入 `PYTHONPATH`。

**Q2：提示 ReID 权重 / YOLO 权重文件不存在？**

检查 `reid/osnet_x0_25_imagenet.pth` 与 `runs/detect/<实验名>/weights/best.pt` 是否存在，并在 `gggggg.py` / 评估脚本中修改为实际路径。

**Q3：没有 GPU 可以运行吗？**

可以。系统会自动检测 CUDA，无 GPU 时回退到 CPU，但推理速度会明显下降。

**Q4：训练脚本中的路径报错？**

脚本与 `*.yaml` 中使用了绝对路径，迁移到其他机器后请统一替换为实际路径。

---

## 参考与致谢

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Torchreid / OSNet](https://github.com/KaiyangZhou/deep-person-reid)
- [MOT Challenge](https://motchallenge.net/)
- [CrowdHuman](https://www.crowdhuman.org/)

---

## 许可证

- 本项目代码仅供学习与研究使用。
- `ultralytics`（YOLO11）遵循 AGPL-3.0 许可。
- `deep_sort/` 子目录遵循其原有 LICENSE 协议。
- CrowdHuman（Roboflow 导出版）遵循 CC BY 4.0。
