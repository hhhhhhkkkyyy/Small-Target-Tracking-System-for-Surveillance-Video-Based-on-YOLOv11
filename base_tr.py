# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     PATH = r'D:\666\PyCharm 2023.3.2\pythonProject\dadada\yolo11n.pt'
#     model = YOLO(PATH)  # 或你的本地 yolo11n.pt
#
#     model.train(
#         data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\crowdhuman\data.yaml",
#         epochs=50,
#         imgsz=800,              # CrowdHuman 最佳
#         batch=8,               # 你的GPU轻松跑
#         optimizer="AdamW",
#         lr0=0.001,
#         patience=20,
#         pretrained=True,
#         mosaic=1.0,
#         mixup=0.1,
#         copy_paste=0.3,
#         close_mosaic=10,
#         single_cls=True,
#         name="crowdhuman_yolov11n_baseline",
#         device=0,
#         workers=4,
#         cache=False,
#     )

from ultralytics import YOLO

if __name__ == '__main__':
    # 核心修改1：替换为yolo11s预训练权重（本地无则自动下载）
    PATH = r'D:\666\PyCharm 2023.3.2\pythonProject\dadada\yolo11s.pt'
    model = YOLO(PATH)  # 本地无yolo11s.pt时，直接写YOLO("yolo11s.pt")会自动下载

    model.train(
        data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\crowdhuman\data.yaml",
        epochs=50,  # 11s参数量更大，50轮足够充分训练（若想更快收敛可设40）
        imgsz=800,  # 保持800不变，适配CrowdHuman数据集特性
        batch=8,    # RTX4060 8GB显存可稳定运行（11s batch=8无压力）
        optimizer="AdamW",
        lr0=0.0008,  # 核心修改2：比11n的0.001略低，避免11s过拟合（参数量翻倍）
        patience=20, # 早停耐心值保持，11s收敛稍慢，留足调整空间
        pretrained=True,  # 启用11s官方预训练权重，加速收敛
        mosaic=1.0,
        mixup=0.2,   # 核心修改3：增强mixup（11s抗过拟合能力更强）
        copy_paste=0.4,  # 核心修改4：提高copy_paste比例，强化密集行人场景学习
        close_mosaic=20, # 核心修改5：延迟关闭mosaic（前20轮），充分学习密集特征
        single_cls=True,
        name="crowdhuman_yolov11s_baseline",  # 核心修改6：修改实验名，区分11n基线
        device=0,
        workers=5,
        cache=True,  # 核心修改7：开启缓存，抵消11s计算量增加带来的训练速度下降
        # 新增适配11s的关键参数（可选，进一步优化性能）
        box=8.0,     # 提高框损失权重，11s框回归能力更强
        cls=0.4,     # 降低分类损失（单类行人分类简单）
        dfl=1.8,     # 提高分布损失，优化密集场景边界框回归
        conf=0.2,    # 降低置信度阈值，提升11s小目标召回率
        max_det=600, # 增加最大检测数，适配11s更强的密集检测能力
        cos_lr=True, # 启用余弦学习率衰减，更适配11s的微调
        warmup_epochs=5, # 延长热身期，稳定11s初始训练
    )