import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image

# ==========================================
# 1. 全局超参数配置 (Hyperparameters)
# ==========================================
SOURCE_DIR = "/home/ljc/code/PaBoT-main/brain"
TARGET_DIR = "/home/ljc/code/PaBoT-main/datasets" 

TARGET_SHAPE = (256, 256)
SPLIT_RATIOS = {'train': 40, 'val': 5, 'test': 15}

# 组织占比阈值：若切片中有效组织的像素少于此比例，则丢弃
TISSUE_THRESHOLD_RATIO = 0.05 

# ==========================================
# 2. 核心数学预处理算子 (Preprocessing Operators)
# ==========================================
def pad_to_target_shape(image_2d, target_shape=TARGET_SHAPE):
    """
    执行独立维度的居中零填充 (Center Zero-Padding) 或中心裁剪 (Center Cropping)。
    注意：此函数现在接收的是已经映射为 uint8 的矩阵，填充的 0 即为绝对纯黑。
    """
    h, w = image_2d.shape
    th, tw = target_shape
    
    # 1. 解耦处理 H 维度: 越界则中心裁剪，否则计算填充余量
    if h > th:
        start_h = (h - th) // 2
        image_2d = image_2d[start_h : start_h + th, :]
        pad_h_top, pad_h_bot = 0, 0
    else:
        pad_h_top = (th - h) // 2
        pad_h_bot = th - h - pad_h_top
        
    # 2. 解耦处理 W 维度: 越界则中心裁剪，否则计算填充余量
    h, w = image_2d.shape  # 重新获取可能被裁剪后的维度
    if w > tw:
        start_w = (w - tw) // 2
        image_2d = image_2d[:, start_w : start_w + tw]
        pad_w_left, pad_w_right = 0, 0
    else:
        pad_w_left = (tw - w) // 2
        pad_w_right = tw - w - pad_w_left
        
    # 3. 仅当存在低于目标维度的轴时，执行 Padding 算子
    if pad_h_top > 0 or pad_h_bot > 0 or pad_w_left > 0 or pad_w_right > 0:
        image_2d = np.pad(
            image_2d, 
            ((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)), 
            mode='constant', 
            constant_values=0
        )
        
    return image_2d

def normalize_ct(ct_array, window_min=-1000, window_max=1000):
    """ CT: 刚性物理截断 (HU Windowing) """
    clipped = np.clip(ct_array, window_min, window_max)
    normalized = (clipped - window_min) / (window_max - window_min)
    return (normalized * 255.0).astype(np.uint8)

def normalize_mr(mr_array, val_min, val_max):
    """ MR: 基于全局 3D 统计的百分位映射，避免切片间的亮度闪烁 """
    if val_max - val_min < 1e-6:
        return np.zeros_like(mr_array, dtype=np.uint8)
        
    clipped = np.clip(mr_array, val_min, val_max)
    normalized = (clipped - val_min) / (val_max - val_min)
    return (normalized * 255.0).astype(np.uint8)

def has_enough_tissue(mr_slice, threshold_ratio=TISSUE_THRESHOLD_RATIO):
    """ 计算切片信息熵：简单地通过像素强度阈值估算组织面积 """
    threshold_val = np.max(mr_slice) * 0.1
    tissue_area = np.sum(mr_slice > threshold_val)
    total_area = mr_slice.size
    return (tissue_area / total_area) >= threshold_ratio

# ==========================================
# 3. 单病例处理流水线 (Pipeline per Patient)
# ==========================================
def process_patient(patient_path, target_A_dir, target_B_dir, pt_id):
    mr_file = os.path.join(patient_path, 'mr.nii.gz')
    ct_file = os.path.join(patient_path, 'ct.nii.gz')
    
    if not (os.path.exists(mr_file) and os.path.exists(ct_file)):
        print(f"Warning: {pt_id} 数据不完整，跳过。")
        return 0, 0

    # 加载张量数据
    mr_vol = nib.load(mr_file).get_fdata()
    ct_vol = nib.load(ct_file).get_fdata()
    
    # 空间拓扑等价性断言
    if mr_vol.shape != ct_vol.shape:
        print(f"Error: 空间拓扑不匹配 {pt_id}. MR: {mr_vol.shape}, CT: {ct_vol.shape}")
        return 0, 0
    
    # 计算 MRI 3D 全局极值，保证流形空间的绝对比例连续性
    mr_p_min = np.percentile(mr_vol, 0.5)
    mr_p_max = np.percentile(mr_vol, 99.5)

    depth = mr_vol.shape[2]
    saved_A_count = 0
    saved_B_count = 0
    
    for z in range(depth):
        mr_slice = mr_vol[:, :, z]
        ct_slice = ct_vol[:, :, z]
        
        # 背景过滤
        if not has_enough_tissue(mr_slice):
            continue
            
        # 旋转 90 度以符合常规阅片方向
        mr_slice = np.rot90(mr_slice)
        ct_slice = np.rot90(ct_slice)
        
        # ---------------------------------------------------------
        # 修正核心: 算子顺序倒置
        # 1. 数值映射 (Intensity Mapping): 映射到 0~255 的 uint8 空间
        mr_img_uint8 = normalize_mr(mr_slice, mr_p_min, mr_p_max)
        ct_img_uint8 = normalize_ct(ct_slice, window_min=-1000, window_max=1000)
        
        # 2. 空间对齐 (Spatial Alignment): 此时填充的 0 就是完美的黑色背景
        mr_padded = pad_to_target_shape(mr_img_uint8)
        ct_padded = pad_to_target_shape(ct_img_uint8)
        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        # 域交换映射 (Domain Swap Mapping)
        # 约束条件: Domain A <- CT, Domain B <- MRI
        # ---------------------------------------------------------
        filename = f"{pt_id}_slice{z:03d}.png"
        
        # 记录 I/O 状态，确保统计与实际物理写入对齐
        Image.fromarray(ct_padded).save(os.path.join(target_A_dir, filename))
        saved_A_count += 1
        
        Image.fromarray(mr_padded).save(os.path.join(target_B_dir, filename))
        saved_B_count += 1
        
    print(f"[{pt_id}] 提取完成，有效切片数: A={saved_A_count}, B={saved_B_count} / {depth}")
    return saved_A_count, saved_B_count

# ==========================================
# 4. 主控函数 (Main Controller)
# ==========================================
def main():
    # 初始化文件系统拓扑
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_DIR, f"{split}A"), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, f"{split}B"), exist_ok=True)

    # 提取所有 1BA 开头的病例并排序
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(SOURCE_DIR, "1BA*")) 
        if os.path.isdir(d)
    ])
    
    total_needed = sum(SPLIT_RATIOS.values())
    if len(patient_dirs) < total_needed:
        print(f"Error: 找到的 1BA 病例数为 {len(patient_dirs)}，不足所需的 {total_needed} 例。")
        return

    # 精确截取前 60 例
    patient_dirs = patient_dirs[:total_needed]

    # 构建分配标签
    splits = []
    splits.extend(['train'] * SPLIT_RATIOS['train'])
    splits.extend(['val'] * SPLIT_RATIOS['val'])
    splits.extend(['test'] * SPLIT_RATIOS['test'])

    print(f"开始处理，目标根目录: {TARGET_DIR}")
    print("-" * 50)
    
    # 嵌套状态累加器，解耦 A 域和 B 域的计数
    slice_counts = {
        'train': {'A': 0, 'B': 0},
        'val':   {'A': 0, 'B': 0},
        'test':  {'A': 0, 'B': 0}
    }

    # 迭代计算与分发
    for idx, (p_dir, split_type) in enumerate(zip(patient_dirs, splits)):
        pt_id = os.path.basename(p_dir)
        target_A = os.path.join(TARGET_DIR, f"{split_type}A")
        target_B = os.path.join(TARGET_DIR, f"{split_type}B")
        
        valid_A, valid_B = process_patient(p_dir, target_A, target_B, pt_id)
        
        slice_counts[split_type]['A'] += valid_A
        slice_counts[split_type]['B'] += valid_B

    print("-" * 50)
    print("=== 数据集生成切片统计 (Domain Cardinality) ===")
    
    total_A = 0
    total_B = 0
    for split_type in ['train', 'val', 'test']:
        count_A = slice_counts[split_type]['A']
        count_B = slice_counts[split_type]['B']
        total_A += count_A
        total_B += count_B
        print(f"{split_type.capitalize():<6} -> Domain A (CT): {count_A:<6} | Domain B (MR): {count_B:<6}")
        
    print("-" * 50)
    print(f"Total  -> Domain A (CT): {total_A:<6} | Domain B (MR): {total_B:<6}")
    print("-" * 50)
    
    # 结构完整性断言
    if total_A == total_B:
        print("状态校验: 域间切片数量严格一致，映射流形无结构坍塌。")
    else:
        print("状态校验: 警告！域间切片数量存在偏差，请排查 I/O 持久化错误。")

if __name__ == "__main__":
    main()