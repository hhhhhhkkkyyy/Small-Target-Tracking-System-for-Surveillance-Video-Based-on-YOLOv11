# from ultralytics import YOLO
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
# # 1. 定义SIoU损失（保留）
# class SIoULoss(nn.Module):
#     def __init__(self, eps=1e-7):
#         super().__init__()
#         self.eps = eps
#
#     def forward(self, pred, target):
#         # 转换pred为xyxy（YOLOv11输出格式：(batch, num_anchors, 4+cls+...)）
#         pred_box = pred[..., :4]  # 取框坐标
#         target_box = target[..., :4]
#
#         # 计算IoU
#         x1 = torch.max(pred_box[..., 0], target_box[..., 0])
#         y1 = torch.max(pred_box[..., 1], target_box[..., 1])
#         x2 = torch.min(pred_box[..., 2], target_box[..., 2])
#         y2 = torch.min(pred_box[..., 3], target_box[..., 3])
#         inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
#         union = (pred_box[..., 2]-pred_box[..., 0])*(pred_box[..., 3]-pred_box[..., 1]) + \
#                 (target_box[..., 2]-target_box[..., 0])*(target_box[..., 3]-target_box[..., 1]) - inter + self.eps
#         iou = inter / union
#
#         # SIoU角度/距离/形状损失
#         cx_p = (pred_box[..., 0] + pred_box[..., 2]) / 2
#         cy_p = (pred_box[..., 1] + pred_box[..., 3]) / 2
#         cx_t = (target_box[..., 0] + target_box[..., 2]) / 2
#         cy_t = (target_box[..., 1] + target_box[..., 3]) / 2
#         dx = cx_t - cx_p
#         dy = cy_t - cy_p
#         theta = torch.atan2(dy, dx + self.eps)
#         angle_cost = 1 - 2 * torch.sin(torch.abs(theta) - math.pi/4) **2
#
#         rho = torch.sqrt(dx**2 + dy**2)
#         c = torch.sqrt((target_box[..., 2]-target_box[..., 0] + pred_box[..., 2]-pred_box[..., 0])**2 +
#                        (target_box[..., 3]-target_box[..., 1] + pred_box[..., 3]-pred_box[..., 1])**2) / 2
#         distance_cost = 1 - torch.exp(-rho/(c + self.eps))
#
#         w_p = pred_box[..., 2] - pred_box[..., 0]
#         h_p = pred_box[..., 3] - pred_box[..., 1]
#         w_t = target_box[..., 2] - target_box[..., 0]
#         h_t = target_box[..., 3] - target_box[..., 1]
#         shape_cost = torch.abs(1 - torch.sqrt(w_t/(w_p+self.eps))) + torch.abs(1 - torch.sqrt(h_t/(h_p+self.eps)))
#
#         siou = iou - (angle_cost + distance_cost + shape_cost)/3
#         return (1 - siou).mean()
#
# # 2. 复用GAM+小目标分支（同方案1）
# class GAM_SmallTargetFusion(nn.Module):
#     def __init__(self, channel):
#         super().__init__()
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel//8, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel//8, channel, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.sa = nn.Sequential(
#             nn.Conv2d(2, 1, 3, padding=1, bias=False),
#             nn.Sigmoid()
#         )
#         self.small_branch = nn.Sequential(
#             nn.Conv2d(channel, channel//2, 1, bias=False),
#             nn.BatchNorm2d(channel//2),
#             nn.ReLU(),
#             nn.Conv2d(channel//2, channel, 1, bias=False),
#         )
#         self.fusion_weight = nn.Parameter(torch.ones(2), requires_grad=True)
#
#     def forward(self, x):
#         residual = x
#         ca = self.ca(x)
#         sa = self.sa(torch.cat([torch.mean(x,1,keepdim=True), torch.max(x,1,keepdim=True)[0]],1))
#         x_gam = x * ca * sa
#         x_small = self.small_branch(x)
#         x = residual + self.fusion_weight[0]*x_gam + self.fusion_weight[1]*x_small
#         return x
#
# # 3. 重写模型损失计算（核心：替换box损失）
# def override_model_loss(model, siou_loss):
#     # 保存原始前向损失函数
#     original_loss = model.model.loss
#     # 重写损失函数
#     def custom_loss(preds, batch):
#         loss_dict = original_loss(preds, batch)
#         # 替换box损失为SIoU
#         pred_box = preds[0][..., :4]  # 取第一个输出的框坐标
#         target_box = batch['bboxes']  # 目标框
#         # 适配维度（batch, num_boxes, 4）
#         if len(pred_box.shape) == 4:
#             pred_box = pred_box.permute(0, 2, 3, 1).reshape(-1, 4)
#         if len(target_box.shape) == 3:
#             target_box = target_box.reshape(-1, 4)
#         # 计算SIoU损失并替换
#         siou_box_loss = siou_loss(pred_box, target_box)
#         loss_dict['box'] = siou_box_loss * 9.0  # 应用box权重
#         return loss_dict
#     # 绑定自定义损失
#     model.model.loss = custom_loss
#     print("✅ 重写损失函数，替换为SIoU")
#
# # 4. 主训练逻辑
# if __name__ == '__main__':
#     best_weight = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\crowdhuman_yolov11s_innov1_gam_optim2\weights\best.pt"
#     model = YOLO(best_weight)
#     model.model = model.model.to('cuda:0')
#
#     # 插入GAM模块
#     target_layer = 9
#     layer = model.model.model[target_layer]
#     ch = 512
#     fusion_module = GAM_SmallTargetFusion(ch).to('cuda:0')
#     model.model.model[target_layer] = nn.Sequential(layer, fusion_module)
#
#     # 初始化SIoU损失并覆盖模型损失
#     siou_loss = SIoULoss().to('cuda:0')
#     override_model_loss(model, siou_loss)
#
#     # 训练（参数同方案1）
#     results = model.train(
#         data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\crowdhuman\data.yaml",
#         epochs=50,
#         resume=False,
#         imgsz=800,
#         batch=8,
#         optimizer="AdamW",
#         lr0=0.00008,
#         lrf=0.0008,
#         warmup_epochs=4,
#         mosaic=0.9,
#         mixup=0.0,
#         copy_paste=0.55,
#         close_mosaic=8,
#         conf=0.18,
#         max_det=800,
#         iou=0.48,
#         box=9.0,
#         cls=0.45,
#         dfl=2.2,
#         patience=80,
#         pretrained=False,
#         single_cls=True,
#         name="crowdhuman_yolov11s_gam_siou1",
#         device=0,
#         workers=4,
#         cache=True,
#         amp=True,
#         val=True,
#         save=True,
#         plots=True,
#     )
#
#     # 输出结果
#     print("\n===== 创新点一+自定义SIoU 最终结果 =====")
#     if hasattr(results, 'results_dict'):
#         print(f"mAP50：{results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
#         print(f"召回率：{results.results_dict.get('metrics/recall(B)', 0):.4f}")
#         print(f"精确率：{results.results_dict.get('metrics/precision(B)', 0):.4f}")
#     else:
#         print(f"模型保存路径：{results.save_dir}")


from ultralytics import YOLO
import torch
import torch.nn as nn
import math

# =========================
# 1. 自定义 SIoU Loss
# =========================
class SIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: [N, 4] -> xyxy
        px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area_p = (px2 - px1) * (py2 - py1)
        area_t = (tx2 - tx1) * (ty2 - ty1)
        union = area_p + area_t - inter + self.eps
        iou = inter / union

        # center distance
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2

        dx = tcx - pcx
        dy = tcy - pcy
        rho = torch.sqrt(dx ** 2 + dy ** 2)

        cw = (tx2 - tx1 + px2 - px1) / 2
        ch = (ty2 - ty1 + py2 - py1) / 2
        c = torch.sqrt(cw ** 2 + ch ** 2) + self.eps

        distance_cost = 1 - torch.exp(-rho / c)

        wp = px2 - px1
        hp = py2 - py1
        wt = tx2 - tx1
        ht = ty2 - ty1

        shape_cost = torch.abs(1 - torch.sqrt(wt / (wp + self.eps))) + \
                     torch.abs(1 - torch.sqrt(ht / (hp + self.eps)))

        siou = iou - (distance_cost + shape_cost) / 2
        return (1 - siou).mean()


# =========================
# 2. GAM + 小目标增强模块
# =========================
class GAM_SmallTargetFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.small_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False)
        )

        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        residual = x
        ca = self.ca(x)
        sa = self.sa(
            torch.cat([torch.mean(x, 1, keepdim=True),
                       torch.max(x, 1, keepdim=True)[0]], dim=1)
        )
        x_gam = x * ca * sa
        x_small = self.small_branch(x)
        return residual + self.w[0] * x_gam + self.w[1] * x_small


# =========================
# 3. 覆盖 YOLO box loss → SIoU
# =========================
def override_loss(model, siou_loss):
    original_loss = model.model.loss

    def custom_loss(preds, batch):
        loss_dict = original_loss(preds, batch)

        pred_box = preds[0][..., :4].reshape(-1, 4)
        target_box = batch["bboxes"].reshape(-1, 4)

        loss_dict["box"] = siou_loss(pred_box, target_box) * 9.0
        return loss_dict

    model.model.loss = custom_loss
    print("✅ Box Loss 已替换为 SIoU")


# =========================
# 4. 主训练入口（直接 MOT17）
# =========================
if __name__ == "__main__":
    device = "cuda:0"

    # ⚠️ 只使用 YOLOv11s 原始权重（不是 CrowdHuman）
    PATH = r'D:\666\PyCharm 2023.3.2\pythonProject\dadada\yolo11s.pt'
    model = YOLO(PATH)
    model.model = model.model.to(device)

    # 插入 GAM + 小目标模块（中高层特征）
    target_layer_idx = 9
    base_layer = model.model.model[target_layer_idx]
    model.model.model[target_layer_idx] = nn.Sequential(
        base_layer,
        GAM_SmallTargetFusion(channels=512)
    ).to(device)

    # SIoU loss
    siou_loss = SIoULoss().to(device)
    override_loss(model, siou_loss)

    # =========================
    # 直接 MOT17 训练
    # =========================
    model.train(
        data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo\data.yaml",
        epochs=50,
        imgsz=800,
        batch=8,
        optimizer="AdamW",
        lr0=8e-5,
        lrf=8e-4,
        warmup_epochs=4,

        mosaic=0.0,        # 视频场景，关闭
        mixup=0.0,
        copy_paste=0.0,

        box=9.0,
        cls=0.45,
        dfl=2.2,

        single_cls=True,
        pretrained=False,
        patience=80,
        device=0,
        workers=4,
        cache=True,
        amp=True,
        val=True,

        name="mot17_yolov11s_gam_siou",
        save=True,
        plots=True
    )
# from ultralytics import YOLO
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # 优化1：轻量化GAM（适配YOLOv11s，增加残差+动态通道）
# class LightGAM(nn.Module):
#     def __init__(self, channel, ratio=8):
#         super().__init__()
#         # 通道注意力：简化卷积，增加BN提升稳定性
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.BatchNorm2d(channel // ratio),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False),
#             nn.BatchNorm2d(channel),
#             nn.Sigmoid()
#         )
#         # 空间注意力：缩小卷积核（7→3），减少计算+避免过拟合
#         self.sa = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 3, padding=1, bias=False),
#             nn.BatchNorm2d(channel // ratio),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // ratio, 1, 3, padding=1, bias=False),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         # 残差连接：保留原始特征，仅增强不破坏
#         self.residual_weight = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # 可学习残差权重
#
#     def forward(self, x):
#         residual = x
#         x = x * self.ca(x)
#         x = x * self.sa(x)
#         # 残差融合：平衡增强特征与原始特征
#         x = residual + self.residual_weight * x
#         return x
#
# # 优化2：精准解析层通道数（避免手动设定错误）
# def get_layer_out_channels(layer):
#     """递归解析YOLOv11s任意层的输出通道数"""
#     if isinstance(layer, nn.Conv2d):
#         return layer.out_channels
#     elif isinstance(layer, nn.Sequential):
#         return get_layer_out_channels(layer[-1])
#     elif hasattr(layer, 'model'):
#         return get_layer_out_channels(layer.model[-1])
#     else:
#         return 128  # 兜底值
#
# # 优化3：适配YOLOv11s的GAM插入逻辑
# def add_optimized_gam(model):
#     # 修正插入位置：选择小目标特征更丰富的中层（8/11层），避开高层语义层
#     targets = [8, 11]  # 替换原13/19层，适配YOLOv11s特征金字塔
#     count = 0
#     device = next(model.model.parameters()).device  # 自动匹配模型设备
#     for i in targets:
#         if i < len(model.model.model):
#             layer = model.model.model[i]
#             # 精准获取层输出通道，避免手动设定错误
#             channel = get_layer_out_channels(layer)
#             gam = LightGAM(channel=channel).to(device)
#             model.model.model[i] = nn.Sequential(layer, gam)
#             print(f"✅ 插入LightGAM到层{i}，通道数：{channel}")
#             count += 1
#     print(f"创新1完成：插入 {count} 个优化版GAM模块")
#
# if __name__ == '__main__':
#     baseline_pt = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\crowdhuman_yolov11s_baseline\weights\best.pt"
#     model = YOLO(baseline_pt)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model.model = model.model.to(device)
#
#     # 插入优化版GAM
#     add_optimized_gam(model)
#
#     # 优化4：适配GAM的保守训练策略（解决30轮mAP停滞）
#     model.train(
#         data=r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\crowdhuman\data.yaml",
#         epochs=50,
#         imgsz=800,
#         batch=8,
#         optimizer="AdamW",
#         lr0=0.0002,         # 降低学习率（0.0003→0.0002），避免梯度爆炸
#         lrf=0.001,          # 放缓学习率衰减，30轮后仍有有效更新
#         patience=20,
#         pretrained=False,
#         # 数据增强：针对性强化密集小目标（适配GAM注意力）
#         mosaic=1.0,
#         mixup=0.18,         # 适度提升mixup，增加遮挡场景注意力学习
#         copy_paste=0.4,     # 提升copy_paste，强化密集行人特征
#         close_mosaic=20,    # 延迟关闭mosaic，让GAM学习多尺度特征
#         # 检测参数：适配小目标，避免增强特征被过滤
#         single_cls=True,
#         name="crowdhuman_yolov11s_innov1_gam_optim2",
#         device=0,
#         workers=4,
#         cache=True,         # 开启cache加速训练，避免数据加载拖慢收敛
#         amp=True,
#         val=True,
#         save=True,
#         plots=True,
#         iou=0.52,           # 降低NMS阈值，保留GAM增强的小目标检测框
#         conf=0.22,          # 降低置信度阈值，避免小目标漏检
#         max_det=600,        # 增加最大检测数，适配密集场景
#         # 损失权重：强化框回归，与GAM注意力协同
#         box=8.0,
#         cls=0.45,
#         dfl=1.8,
#     )
#
#     # 输出关键结果
#     print("\n===== 优化版GAM训练结果 =====")
#     print(f"mAP50：{model.results.results_dict['metrics/mAP50(B)']:.4f}")
#     print(f"召回率：{model.results.results_dict['metrics/recall(B)']:.4f}")
#     print(f"精确率：{model.results.results_dict['metrics/precision(B)']:.4f}")
