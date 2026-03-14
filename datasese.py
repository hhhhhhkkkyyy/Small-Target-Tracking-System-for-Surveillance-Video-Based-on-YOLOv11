import shutil
from pathlib import Path

# ===================== 配置区 =====================
MOT_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17"
SAVE_ROOT = r"D:\666\PyCharm 2023.3.2\pythonProject\dadada\MOT17_yolo"
DETECTOR_FILTER = "FRCNN"  # 仅处理FRCNN检测器的序列
# 创建submit文件夹路径（最终存放重命名后的txt文件）
SUBMIT_DIR = Path(SAVE_ROOT) / "submit"

def main():
    # 1. 创建submit目录（如果不存在，递归创建父目录）
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 提交文件保存目录：{SUBMIT_DIR}")

    # 统计处理的序列数量
    processed_seqs = 0

    # 2. 处理train和test两个split下的FRCNN序列
    for split_type in ["train", "test"]:
        split_path = Path(MOT_ROOT) / split_type
        if not split_path.exists():
            print(f"⚠️ 未找到 {split_type} 目录，跳过")
            continue

        # 3. 筛选仅包含FRCNN的序列目录（匹配MOT17-XX-FRCNN格式）
        seq_dirs = [
            d for d in split_path.iterdir()
            if d.is_dir() and "MOT17-" in d.name and DETECTOR_FILTER in d.name
        ]

        if not seq_dirs:
            print(f"⚠️ {split_type} 目录下无{DETECTOR_FILTER}检测器序列，跳过")
            continue

        # 4. 处理每个FRCNN序列
        for seq_dir in seq_dirs:
            seq_name = seq_dir.name  # 获取序列名（如MOT17-02-FRCNN）
            gt_file_path = seq_dir / "gt" / "gt.txt"  # 原始gt.txt路径

            # 检查gt.txt是否存在
            if not gt_file_path.exists():
                print(f"⚠️ {seq_name} 未找到gt.txt文件，跳过")
                continue

            # 5. 定义目标文件路径：submit/序列名.txt
            target_txt_name = f"{seq_name}.txt"
            target_txt_path = SUBMIT_DIR / target_txt_name

            # 6. 复制gt.txt并重命名为目标文件（覆盖已存在的同名文件）
            shutil.copyfile(gt_file_path, target_txt_path)
            print(f"✅ 处理完成：{seq_name} -> {target_txt_name}")

            processed_seqs += 1

    # 7. 输出最终统计信息
    print("\n" + "=" * 50)
    if processed_seqs > 0:
        print(f"✅ 数据集处理完成！共处理 {processed_seqs} 个FRCNN序列")
        print(f"📊 所有重命名后的文件已存放至：{SUBMIT_DIR}")
    else:
        print(f"❌ 未处理任何FRCNN序列，请检查数据集路径")
    print("=" * 50)

if __name__ == "__main__":
    main()