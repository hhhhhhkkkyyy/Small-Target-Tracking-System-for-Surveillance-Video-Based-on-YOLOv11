# Small-Target-Tracking-System-for-Surveillance-Video-Based-on-YOLOv11
项目概述
本项目基于 Ultralytics YOLOv11s，围绕 CrowdHuman 与 MOT17 行人检测任务，实现了从基线训练到创新改进的完整研究流程。核心贡献包括：
YOLOv11s 基线模型训练 - 在CrowdHuman数据集上建立性能基准
结构创新 - 引入GAM注意力机制+小目标增强分支
损失函数创新 - 自定义SIoU Loss替换传统IoU损失
迁移学习优化 - 从CrowdHuman到MOT17的有效迁移
数据预处理自动化 - MOT17标注文件的智能修复
项目目录结构
Dadada/
│
├── README.md                    # 项目说明文档
│
├── base_tr.py                   # YOLOv11s基线训练（CrowdHuman）
├── bebebe4.py                   # GAM+SIoU改进模型训练（核心创新）
├── qianyi.py                    # MOT17标注修复+迁移训练
│
├── yolo11s.pt                   # YOLOv11s官方预训练权重
│
├── datasets/
│   ├── crowdhuman/              # CrowdHuman数据集
│   │   ├── images/
│   │   ├── labels/
│   │   └── data.yaml
│   │
│   ├── MOT17_yolo/              # MOT17（YOLO格式）
│   │   ├── images/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── data.yaml
│   │
│   └── MOT17/                   # MOT17原始数据
│       ├── train/
│       └── test/
│
├── models/
│   ├── reid/
│   │   └── osnet_x0_25_imagenet.pth  # ReID模型权重
│   └── checkpoints/             # 训练中间权重
│
└── runs/                        # 训练输出目录（自动生成）
    ├── detect/                  # 检测任务
│   ├── crowdhuman_yolov11s_baseline/
│   ├── crowdhuman_yolov11s_innov1_gam_optim2/
│   ├── crowdhuman_yolov11s_gam_siou1/
    │   └── mot17_1_baseline/
    └── track_eval_mot17_2511/   # 跟踪评估结果
环境配置
软件依赖
# 创建Python环境（推荐3.9）
conda create -n yolo11 python=3.9
conda activate yolo11
# 安装PyTorch（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 安装核心依赖
pip install ultralytics==8.3.235
pip install opencv-python==4.12.0
pip install numpy==1.25.2
pip install motmetrics==1.4.0
pip install scikit-learn==1.7.2
pip install matplotlib==3.10.6
pip install tqdm==4.66.1
# 安装跟踪相关
pip install filterpy==1.4.5
pip install scipy==1.11.4
pip install lap==0.4.0
核心代码说明
1. base_tr.py - 基线模型训练
功能: YOLOv11s在CrowdHuman上的基准训练
运行命令:
python base_tr.py
2. bebebe4.py - 创新模型训练（核心）
核心创新点:
GAM注意力机制 - 通道+空间注意力增强
小目标增强分支 - 专门提升小目标检测
SIoU损失函数 - 替换传统IoU，优化边界框回归
python bebebe4.py
3. qianyi.py - MOT17数据预处理与迁移训练
主要功能:
自动标注修复 - 处理坐标越界、负值等问题
迁移学习训练 - 从CrowdHuman到MOT17
运行命令:

bash
python qianyi.py
4. test.py - MOT17跟踪测试评估
主要功能:
根据权重和reid来进行评估
运行命令:
python test.py
