import tkinter as tk
from collections import defaultdict
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torchreid
from torchvision import transforms

# DeepSORT 相关导入
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
import warnings, torchreid
warnings.filterwarnings("ignore", message="Cython evaluation.*")
# ===================== 配置项（可统一修改）=====================
# 模型路径（根据你的实际路径调整）
MODEL_PATHS = {
    "YOLO11s Baseline（从头训练）":
        r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_yolov11s_from_scratch\weights\best.pt",

    "YOLO11s + GAM + 小目标增强 + SIoU（改进模型）":
        r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_1_baseline\weights\best.pt"
}

# ReID 优化配置
REID_CONF_THRESH = 0.6  # 低于该置信度才计算 ReID
REID_FRAME_INTERVAL = 3  # 隔N帧计算 ReID
MAX_REID_NUM = 10  # 每帧最大 ReID 计算数量
FEATURE_DIM = 512  # ReID 特征维度

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DeepSORT 配置
MAX_COSINE_DISTANCE = 0.4
NN_BUDGET = 100
MAX_AGE = 60  # 最大遮挡帧数
N_INIT = 3  # 确认轨迹所需初始帧数


# ===================== 修复 DeepSORT 距离计算逻辑 =====================
class FixedNearestNeighborDistanceMetric(nn_matching.NearestNeighborDistanceMetric):

    def distance(self, features, targets):
        num_t = len(targets)
        num_f = len(features)

        # 默认最大代价
        cost_matrix = np.ones((num_t, num_f), dtype=np.float32)

        if num_t == 0 or num_f == 0:
            return cost_matrix

        # ========= 处理 detection features =========
        valid_feat_idx = []
        valid_feats = []

        for fi, feat in enumerate(features):
            # 1️⃣ None 直接跳过
            if feat is None:
                continue

            # 2️⃣ 必须是 ndarray，list 一律丢弃（⚠️ 关键）
            if not isinstance(feat, np.ndarray):
                continue

            # 3️⃣ 必须是 1D 且维度正确
            if feat.ndim != 1 or feat.shape[0] != FEATURE_DIM:
                continue

            # 4️⃣ 非零范数
            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                continue

            valid_feats.append(feat / norm)
            valid_feat_idx.append(fi)

        if len(valid_feats) == 0:
            return cost_matrix

        valid_feats = np.stack(valid_feats, axis=0)

        # ========= 处理 track features =========
        valid_tgt_idx = []
        valid_targets = []

        for ti, track_id in enumerate(targets):
            if track_id not in self.samples:
                continue

            feat = self.samples[track_id]

            if feat is None:
                continue

            # ⚠️ 同样：不是 ndarray 的直接丢弃
            if not isinstance(feat, np.ndarray):
                continue

            if feat.ndim != 1 or feat.shape[0] != FEATURE_DIM:
                continue

            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                continue

            valid_targets.append(feat / norm)
            valid_tgt_idx.append(ti)

        if len(valid_targets) == 0:
            return cost_matrix

        valid_targets = np.stack(valid_targets, axis=0)

        # ========= 计算 cosine distance =========
        similarity = np.dot(valid_targets, valid_feats.T)
        cost = 1.0 - similarity

        # 写回总代价矩阵
        for i, ti in enumerate(valid_tgt_idx):
            for j, fi in enumerate(valid_feat_idx):
                cost_matrix[ti, fi] = cost[i, j]

        return cost_matrix


# ===================== 主 GUI 类 =====================
class MOTTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("基于 YOLO11 的监控视频小目标跟踪系统")
        self.root.geometry("1200x720")
        self.root.configure(bg="#F2F2F2")
        self.root.resizable(False, False)

        # 核心状态变量
        self.video_path = None
        self.running = False
        self.cap = None
        self.latest_frame = None
        self.model = None
        self.video_writer = None

        # 模型选择
        self.current_model_name = tk.StringVar(value=list(MODEL_PATHS.keys())[0])

        # 帧计数（用于 ReID 降频）
        self.frame_id = 0

        # 跟踪历史
        self.track_history = defaultdict(list)

        # 统计量（线程安全）
        self.det_count = 0
        self.track_count = 0
        self.fps = 0
        self.prev_time = time.time()

        # 模式切换（实时/高精度）
        self.mode_var = tk.StringVar(value="高精度模式")
        self.real_time_mode = False

        # 初始化 UI
        self.build_ui()

        # 定时刷新 UI（非阻塞）
        self.root.after(30, self.update_ui)

        # 初始化 DeepSORT（使用修复版度量）
        self.init_deepsort()

        # 初始化 ReID 模型
        self.init_reid_model()

    # ===================== 初始化模块 =====================
    def init_deepsort(self):
        """初始化 DeepSORT 跟踪器（使用修复版距离度量）"""
        metric = FixedNearestNeighborDistanceMetric(
            "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
        )
        self.tracker = Tracker(
            metric,
            max_age=MAX_AGE,
            n_init=N_INIT
        )

    def init_reid_model(self):
        """初始化 ReID 模型（OSNet-x0.25）"""
        # 构建模型
        self.reid_model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1000,
            loss='softmax',
            pretrained=False
        )

        # 加载权重
        try:
            state_dict = torch.load(
                r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\reid\osnet_x0_25_imagenet.pth",
                map_location=DEVICE
            )
            self.reid_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            messagebox.showerror("ReID 模型加载失败", f"错误：{str(e)}")
            self.reid_model = None

        # 设置为评估模式
        if self.reid_model:
            self.reid_model.to(DEVICE).eval()

        # ReID 图像变换
        self.reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # ===================== ReID 特征提取（核心修复）=====================
    @torch.no_grad()
    def extract_reid_feature(self, img):
        """
        提取 ReID 特征（鲁棒版，确保维度正确）
        :param img: BGR 格式的裁剪图像
        :return: 512维归一化特征向量，失败返回None
        """
        # 空图像直接返回 None
        if img is None or img.size == 0 or self.reid_model is None:
            return None

        try:
            # 转换为 RGB 并预处理
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.reid_transform(img_rgb).unsqueeze(0).to(DEVICE)

            # 提取特征
            feat = self.reid_model(tensor)
            feat = feat.cpu().numpy().reshape(-1)

            # 确保特征维度正确
            if len(feat) != FEATURE_DIM:
                feat = np.zeros(FEATURE_DIM, dtype=np.float32)

            # 安全归一化（避免除以 0）
            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                return None

            # 返回单位向量
            return feat / norm

        except Exception as e:
            print(f"ReID 特征提取失败：{e}")
            return None

    # ===================== UI 构建 =====================
    def build_ui(self):
        """构建完整 UI"""
        # 标题栏
        title = tk.Label(
            self.root,
            text="监控视频小目标检测与跟踪系统（YOLO11 + DeepSORT）",
            font=("微软雅黑", 20, "bold"),
            bg="#1F4E79",
            fg="white",
            height=2
        )
        title.pack(fill=tk.X)

        # 主容器
        main = tk.Frame(self.root, bg="#F2F2F2")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板
        self.control_frame = tk.LabelFrame(
            main, text="控制面板", font=("微软雅黑", 12), bg="#F9F9F9", width=320
        )
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 视频显示面板
        self.display_frame = tk.LabelFrame(
            main, text="视频显示", font=("微软雅黑", 12), bg="#F9F9F9"
        )
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 视频显示标签
        self.video_label = tk.Label(self.display_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 构建控制组件
        self._build_control_widgets()

    def _build_control_widgets(self):
        """构建控制面板组件"""
        pad = {"padx": 20, "pady": 6}

        # 模型选择
        ttk.Label(self.control_frame, text="模型选择：").pack(**pad)
        model_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.current_model_name,
            values=list(MODEL_PATHS.keys()),
            state="readonly"
        )
        model_combo.pack(fill=tk.X, **pad)
        model_combo.current(0)

        # 置信度阈值
        ttk.Label(self.control_frame, text="检测置信度阈值：").pack(**pad)
        self.conf_var = tk.DoubleVar(value=0.30)
        self.conf_scale = ttk.Scale(
            self.control_frame,
            from_=0.1,
            to=0.9,
            variable=self.conf_var,
            command=self.update_conf_label
        )
        self.conf_scale.pack(fill=tk.X, **pad)
        self.conf_value_label = ttk.Label(
            self.control_frame,
            text="当前置信度：0.30"
        )
        self.conf_value_label.pack(**pad)

        # 模式切换（实时/高精度）
        ttk.Label(self.control_frame, text="运行模式：").pack(**pad)
        mode_frame = tk.Frame(self.control_frame, bg="#F9F9F9")
        mode_frame.pack(fill=tk.X, **pad)
        ttk.Radiobutton(
            mode_frame, text="高精度模式", variable=self.mode_var,
            value="高精度模式", command=self.switch_mode
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_frame, text="实时模式", variable=self.mode_var,
            value="实时模式", command=self.switch_mode
        ).pack(side=tk.LEFT)

        # 功能开关
        self.use_camera = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.control_frame,
            text="使用摄像头实时跟踪",
            variable=self.use_camera
        ).pack(padx=20, pady=4)

        self.save_video = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.control_frame,
            text="导出带ID的结果视频",
            variable=self.save_video
        ).pack(padx=20, pady=4)

        # 操作按钮
        ttk.Button(self.control_frame, text="选择视频", command=self.select_video).pack(fill=tk.X, **pad)
        ttk.Button(self.control_frame, text="开始跟踪", command=self.start_tracking).pack(fill=tk.X, **pad)
        ttk.Button(self.control_frame, text="停止", command=self.stop_tracking).pack(fill=tk.X, **pad)

        # 分隔线
        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=8)

        # 统计信息
        self.fps_label = ttk.Label(self.control_frame, text="FPS: 0")
        self.fps_label.pack(**pad)
        self.det_label = ttk.Label(self.control_frame, text="检测数: 0")
        self.det_label.pack(**pad)
        self.track_label = ttk.Label(self.control_frame, text="跟踪数: 0")
        self.track_label.pack(**pad)

        # 分隔线
        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=8)

        # 状态信息
        self.status_label = ttk.Label(self.control_frame, text="状态：未运行", foreground="blue")
        self.status_label.pack(**pad)

    # ===================== UI 辅助函数 =====================
    def update_conf_label(self, _=None):
        """更新置信度显示标签"""
        self.conf_value_label.config(text=f"当前置信度：{self.conf_var.get():.2f}")

    def switch_mode(self):
        """切换运行模式（实时/高精度）"""
        self.real_time_mode = (self.mode_var.get() == "实时模式")
        mode_text = "实时模式（优先速度）" if self.real_time_mode else "高精度模式（优先ID稳定）"
        self.status_label.config(text=f"状态：已切换至 {mode_text}", foreground="orange")

    # ===================== 视频操作 =====================
    def select_video(self):
        """选择视频文件"""
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if path:
            self.video_path = path
            self.status_label.config(text=f"状态：已选择视频 - {os.path.basename(path)}", foreground="blue")

    def start_tracking(self):
        """启动跟踪（新线程运行）"""
        if self.running:
            messagebox.showinfo("提示", "跟踪已在运行中！")
            return

        # 检查输入源
        if not self.use_camera.get() and not self.video_path:
            messagebox.showwarning("提示", "请选择视频文件或启用摄像头！")
            return

        # 重置状态
        self.running = True
        self.track_history.clear()
        self.frame_id = 0
        self.prev_time = time.time()
        self.status_label.config(text=f"状态：运行中（{self.mode_var.get()}）", foreground="green")

        # 启动跟踪线程（守护线程）
        threading.Thread(target=self.run_tracking, daemon=True).start()

    def stop_tracking(self):
        """停止跟踪"""
        self.running = False
        # 释放资源
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.status_label.config(text="状态：已停止", foreground="red")

    # ===================== 核心跟踪逻辑（终极修复）=====================
    def run_tracking(self):
        """跟踪主逻辑（在独立线程中运行）"""
        # 加载 YOLO 模型
        try:
            model_path = MODEL_PATHS[self.current_model_name.get()]
            self.model = YOLO(model_path)
        except Exception as e:
            messagebox.showerror("模型加载失败", f"YOLO 模型加载错误：{str(e)}")
            self.running = False
            return

        # 重新初始化 DeepSORT
        self.init_deepsort()

        # 打开视频/摄像头
        try:
            self.cap = cv2.VideoCapture(0 if self.use_camera.get() else self.video_path)
            if not self.cap.isOpened():
                raise Exception("无法打开视频/摄像头")
        except Exception as e:
            messagebox.showerror("输入源错误", f"无法打开视频/摄像头：{str(e)}")
            self.running = False
            return

        # 初始化视频写入器
        if self.save_video.get():
            try:
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                save_path = f"output_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            except Exception as e:
                messagebox.showwarning("视频保存警告", f"无法初始化视频保存：{str(e)}")
                self.video_writer = None

        # 主跟踪循环
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 帧计数+1
            self.frame_id += 1

            # YOLO 检测
            results = self.model(
                frame,
                conf=self.conf_var.get(),
                iou=0.5,
                verbose=False,
                device=DEVICE
            )[0]

            # 构建 detections（统一特征格式）
            detections = []
            reid_count = 0

            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()

                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1

                    # 过滤无效框
                    if w <= 0 or h <= 0 or x1 >= x2 or y1 >= y2:
                        continue

                    # 初始化特征（确保格式统一）
                    feature = None
                    crop = frame[y1:y2, x1:x2]

                    # 根据模式决定是否计算 ReID
                    if not self.real_time_mode:
                        # 高精度模式：低置信度目标计算 ReID
                        if score < REID_CONF_THRESH:
                            feature = self.extract_reid_feature(crop)
                    else:
                        # 实时模式：降频+数量限制
                        if (self.frame_id % REID_FRAME_INTERVAL == 0 and
                                reid_count < MAX_REID_NUM and score < REID_CONF_THRESH):
                            feature = self.extract_reid_feature(crop)
                            reid_count += 1

                    # 构建 Detection（确保特征不为 Ragged）
                    detections.append(Detection([x1, y1, w, h], score, feature))

            # 更新统计
            self.det_count = len(detections)

            # DeepSORT 跟踪更新（使用修复版度量）
            self.tracker.predict()
            self.tracker.update(detections)

            # 绘制跟踪结果
            self.track_count = 0
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                self.track_count += 1
                # 获取跟踪框和 ID
                x, y, w, h = map(int, track.to_tlwh())
                tid = track.track_id
                cx, cy = x + w // 2, y + h // 2

                # 记录跟踪历史
                self.track_history[tid].append((cx, cy))
                if len(self.track_history[tid]) > 30:
                    self.track_history[tid].pop(0)

                # 绘制 bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 绘制 ID 标签
                cv2.putText(
                    frame, f"ID {tid}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                # 绘制跟踪轨迹
                for i in range(1, len(self.track_history[tid])):
                    cv2.line(
                        frame, self.track_history[tid][i - 1],
                        self.track_history[tid][i], (255, 0, 0), 2
                    )

            # 计算 FPS
            current_time = time.time()
            self.fps = int(1 / max(current_time - self.prev_time, 1e-6))
            self.prev_time = current_time

            # 绘制 FPS
            cv2.putText(
                frame, f"FPS: {self.fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

            # 保存视频
            if self.video_writer and self.save_video.get():
                self.video_writer.write(frame)

            # 更新最新帧（供 UI 显示）
            self.latest_frame = frame.copy()

        # 循环结束，释放资源
        self.stop_tracking()
        messagebox.showinfo("完成", "跟踪已完成！")

    # ===================== UI 实时更新 =====================
    def update_ui(self):
        """非阻塞更新 UI 显示"""
        # 更新视频帧显示
        if self.latest_frame is not None:
            try:
                # 获取显示区域尺寸
                lw = self.video_label.winfo_width()
                lh = self.video_label.winfo_height()

                if lw > 0 and lh > 0:
                    # 等比例缩放
                    h, w, _ = self.latest_frame.shape
                    scale = min(lw / w, lh / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized_frame = cv2.resize(self.latest_frame, (nw, nh))

                    # 转换为 PIL 图像
                    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    tk_img = ImageTk.PhotoImage(image=pil_img)

                    # 更新显示
                    self.video_label.config(image=tk_img)
                    self.video_label.image = tk_img
            except Exception as e:
                print(f"UI 更新错误：{e}")

        # 更新统计信息
        self.fps_label.config(text=f"FPS: {self.fps}")
        self.det_label.config(text=f"检测数: {self.det_count}")
        self.track_label.config(text=f"跟踪数: {self.track_count}")

        # 定时刷新
        self.root.after(30, self.update_ui)


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 解决高 DPI 显示问题
    try:
        tk.CallWrapper(str, "tk", "scaling", 2.0)
    except:
        pass

    # 禁用 NumPy 弃用警告（可选）
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # 启动 GUI
    root = tk.Tk()
    app = MOTTrackerGUI(root)


    # 窗口关闭处理
    def on_closing():
        app.stop_tracking()
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()