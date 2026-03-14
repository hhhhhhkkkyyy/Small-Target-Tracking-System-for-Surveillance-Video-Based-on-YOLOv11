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

# ===================== 模型路径 =====================
MODEL_PATHS = {
    "YOLOv11s Baseline（从头训练）":
        r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_yolov11s_from_scratch\weights\best.pt",

    "YOLOv11s + GAM + 小目标增强 + SIoU（改进模型）":
        r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\runs\detect\mot17_1_baseline\weights\best.pt"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MOTTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("基于 YOLOv11 的监控视频小目标跟踪系统")
        self.root.geometry("1200x720")
        self.root.configure(bg="#F2F2F2")
        self.root.resizable(False, False)

        self.video_path = None
        self.running = False
        self.cap = None
        self.latest_frame = None
        self.model = None

        self.current_model_name = tk.StringVar(value=list(MODEL_PATHS.keys())[0])

        self.track_history = defaultdict(list)

        # ===== 统计量（线程共享）=====
        self.det_count = 0
        self.track_count = 0
        self.fps = 0
        self.prev_time = time.time()

        self.build_ui()
        self.root.after(30, self.update_ui)

    # ===================== UI =====================
    def build_ui(self):
        title = tk.Label(
            self.root,
            text="监控视频小目标检测与跟踪系统（YOLOv11）",
            font=("微软雅黑", 20, "bold"),
            bg="#1F4E79",
            fg="white",
            height=2
        )
        title.pack(fill=tk.X)

        main = tk.Frame(self.root, bg="#F2F2F2")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.use_camera = tk.BooleanVar(value=False)
        self.save_video = tk.BooleanVar(value=True)

        self.video_writer = None

        self.control_frame = tk.LabelFrame(
            main, text="控制面板", font=("微软雅黑", 12), bg="#F9F9F9", width=320
        )
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.display_frame = tk.LabelFrame(
            main, text="视频显示", font=("微软雅黑", 12), bg="#F9F9F9"
        )
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.display_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self._build_controls()

    def _build_controls(self):
        pad = {"padx": 20, "pady": 6}

        ttk.Label(self.control_frame, text="模型选择：").pack(**pad)
        ttk.Combobox(
            self.control_frame,
            textvariable=self.current_model_name,
            values=list(MODEL_PATHS.keys()),
            state="readonly"
        ).pack(fill=tk.X, **pad)

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

        ttk.Checkbutton(
            self.control_frame,
            text="使用摄像头实时跟踪",
            variable=self.use_camera
        ).pack(padx=20, pady=4)

        ttk.Checkbutton(
            self.control_frame,
            text="导出带ID的结果视频",
            variable=self.save_video
        ).pack(padx=20, pady=4)

        ttk.Button(self.control_frame, text="选择视频", command=self.select_video).pack(fill=tk.X, **pad)
        ttk.Button(self.control_frame, text="开始跟踪", command=self.start_tracking).pack(fill=tk.X, **pad)
        ttk.Button(self.control_frame, text="停止", command=self.stop_tracking).pack(fill=tk.X, **pad)

        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=8)

        self.fps_label = ttk.Label(self.control_frame, text="FPS: 0")
        self.fps_label.pack(**pad)
        self.det_label = ttk.Label(self.control_frame, text="检测数: 0")
        self.det_label.pack(**pad)
        self.track_label = ttk.Label(self.control_frame, text="跟踪数: 0")
        self.track_label.pack(**pad)

        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=8)

        self.status_label = ttk.Label(self.control_frame, text="状态：未运行", foreground="blue")
        self.status_label.pack(**pad)

    # ===================== 逻辑 =====================
    def update_conf_label(self, _=None):
        self.conf_value_label.config(text=f"当前置信度：{self.conf_var.get():.2f}")

    def select_video(self):
        path = filedialog.askopenfilename(
            title="选择视频", filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if path:
            self.video_path = path
            self.status_label.config(text="状态：已选择视频")

    def start_tracking(self):
        if self.running:
            return

        if not self.use_camera.get() and not self.video_path:
            messagebox.showwarning("提示", "请选择视频或启用摄像头")
            return

        self.running = True
        self.track_history.clear()
        self.status_label.config(text="状态：运行中", foreground="green")

        threading.Thread(target=self.run_tracking, daemon=True).start()

    def stop_tracking(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.status_label.config(text="状态：已停止", foreground="red")

    # ===================== 跟踪线程 =====================
    def run_tracking(self):
        model_path = MODEL_PATHS[self.current_model_name.get()]
        self.model = YOLO(model_path)

        self.cap = cv2.VideoCapture(0 if self.use_camera.get() else self.video_path)

        if self.save_video.get():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = f"runs/output_{int(time.time())}.mp4"
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        else:
            self.video_writer = None

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            result = self.model.track(
                frame,
                persist=True,
                conf=self.conf_var.get(),
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            track_ids = set()
            self.det_count = 0

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                self.det_count = len(boxes)

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    track_ids.add(tid)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self.track_history[tid].append((cx, cy))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    for i in range(1, len(self.track_history[tid])):
                        cv2.line(frame,
                                 self.track_history[tid][i - 1],
                                 self.track_history[tid][i],
                                 (255, 0, 0), 2)

            self.track_count = len(track_ids)

            now = time.time()
            self.fps = int(1 / max(now - self.prev_time, 1e-6))
            self.prev_time = now

            if self.video_writer:
                self.video_writer.write(frame)

            self.latest_frame = frame.copy()

        if self.video_writer:
            self.video_writer.release()
        self.cap.release()

        self.running = False
        self.status_label.config(text="状态：完成", foreground="blue")

    # ===================== UI刷新 =====================
    def update_ui(self):
        if self.latest_frame is not None:
            h, w, _ = self.latest_frame.shape
            lw = self.video_label.winfo_width()
            lh = self.video_label.winfo_height()

            if lw > 0 and lh > 0:
                scale = min(lw / w, lh / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(self.latest_frame, (nw, nh))
                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(img))
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk

        self.fps_label.config(text=f"FPS: {self.fps}")
        self.det_label.config(text=f"检测数: {self.det_count}")
        self.track_label.config(text=f"跟踪数: {self.track_count}")

        self.root.after(30, self.update_ui)


if __name__ == "__main__":
    root = tk.Tk()
    app = MOTTrackerGUI(root)
    root.mainloop()
