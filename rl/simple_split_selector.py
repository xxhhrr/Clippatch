import os
import yaml
import numpy as np
import json
import random
from PIL import Image
from utils.clip_util import get_heatmap
from tqdm import tqdm

# 所有可能的split策略
SPLIT_ACTIONS = [
    (1, 2), (2, 1), (2, 2),
    (1, 3), (3, 1), (3, 3),
    (3, 2), (2, 3), (1, 5), (5, 1)
]

def _read_json_lines(path):
    with open(path, 'r') as f:
        return [json.loads(l.strip()) for l in f if l.strip()]

def _bbox_to_mask(bbox, size=(224,224), orig_wh=(640,480)):
    """将bbox转换为mask"""
    x, y, w, h = bbox
    H, W = size
    ow, oh = orig_wh
    x1 = int(np.clip(x / ow * W, 0, W-1))
    y1 = int(np.clip(y / oh * H, 0, H-1))
    x2 = int(np.clip((x+w) / ow * W, 0, W-1))
    y2 = int(np.clip((y+h) / oh * H, 0, H-1))
    mask = np.zeros(size, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask

def create_patches(heat_shape, split):
    """根据split策略创建patches"""
    full_patch = np.ones(heat_shape, dtype=np.uint8)
    rows, cols = np.where(full_patch)
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    r, c = split
    h_step = (y_max - y_min + 1) // r
    w_step = (x_max - x_min + 1) // c

    patches = []
    for i in range(r):
        for j in range(c):
            sub = np.zeros_like(full_patch, dtype=np.uint8)
            y_start, y_end = y_min + i * h_step, y_min + (i + 1) * h_step
            x_start, x_end = x_min + j * w_step, x_min + (j + 1) * w_step
            sub[y_start:y_end, x_start:x_end] = 1
            if sub.sum() > 0:
                patches.append(sub)
    
    return patches

def calculate_split_score(heat, patches, gt_mask, patch_bonus_weight=0.0):
    """计算某个split策略的得分，纯粹基于IoU"""
    if not patches:
        return 0.0, -1, 0.0
    
    # 计算每个patch的CLIP得分
    clip_scores = []
    for patch in patches:
        score = (heat * patch).sum()
        clip_scores.append(score)
    
    # 找到得分最高的patch
    best_patch_idx = np.argmax(clip_scores)
    best_patch = patches[best_patch_idx]
    
    # 计算最佳patch与ground truth的IoU
    inter = np.logical_and(best_patch, gt_mask).sum()
    union = np.logical_or(best_patch, gt_mask).sum()
    iou = inter / (union + 1e-6)
    
    # 直接使用IoU作为最终得分，不添加patch数量奖励
    final_score = iou
    
    return final_score, best_patch_idx, iou

def find_best_split(img_path, prompt, gt_mask, verbose=False):
    """为给定的图片和文本找到最佳的split策略"""
    if verbose:
        print(f"Processing: {img_path}")
        print(f"Prompt: {prompt}")
    
    # 获取heatmap
    heat = get_heatmap(img_path, prompt)
    
    best_split = None
    best_score = -1
    best_patch_idx = -1
    best_iou = -1
    results = []
    
    # 尝试每种split策略
    for split in SPLIT_ACTIONS:
        patches = create_patches(heat.shape, split)
        score, patch_idx, iou = calculate_split_score(heat, patches, gt_mask)
        
        results.append({
            'split': split,
            'score': score,
            'iou': iou,
            'best_patch_idx': patch_idx,
            'num_patches': len(patches)
        })
        
        if verbose:
            print(f"Split {split}: Score={score:.4f}, IoU={iou:.4f}, patches={len(patches)}")
        
        if score > best_score:
            best_score = score
            best_split = split
            best_patch_idx = patch_idx
            best_iou = iou
    
    if verbose:
        print(f"\nBest split: {best_split} with Score={best_score:.4f}, IoU={best_iou:.4f}")
    
    return best_split, best_score, results

def process_all_samples(cfg, output_dir="split_results"):
    """处理所有样本并生成带split策略的text文件"""
    image_dir = cfg["data"]["images"]
    text_dir = cfg["data"]["texts"]
    mask_dir = cfg["data"]["masks"]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    processed_count = 0
    failed_count = 0
    split_stats = {}
    
    print(f"开始处理 {len(image_files)} 个图片文件...")
    
    for img_name in tqdm(image_files, desc="Processing images"):
        try:
            # 提取图片ID
            image_id = img_name.split('_')[-1].split('.')[0].zfill(12)
            
            text_path = os.path.join(text_dir, f"{image_id}.txt")
            mask_path = os.path.join(mask_dir, f"{image_id}.txt")
            
            # 检查对应的text和mask文件是否存在
            if not (os.path.exists(text_path) and os.path.exists(mask_path)):
                continue
            
            img_path = os.path.join(image_dir, img_name)
            prompts = _read_json_lines(text_path)
            instances = _read_json_lines(mask_path)
            
            if not prompts or not instances:
                continue
            
            # 获取图片原始尺寸
            with Image.open(img_path) as img:
                orig_wh = img.size
            
            # 创建输出文件
            output_path = os.path.join(output_dir, f"{image_id}.txt")
            
            with open(output_path, 'w', encoding='utf-8') as out_f:
                instance_map = {inst['ann_id']: inst for inst in instances}
                
                for prompt_data in prompts:
                    if prompt_data['ann_id'] not in instance_map:
                        # 如果没有对应的mask，写入原始数据但不添加split
                        out_f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
                        continue
                    
                    ann_data = instance_map[prompt_data['ann_id']]
                    prompt = prompt_data['sent']
                    gt_mask = _bbox_to_mask(ann_data["bbox"], orig_wh=orig_wh)
                    
                    # 找到最佳split策略
                    best_split, best_score, _ = find_best_split(img_path, prompt, gt_mask)
                    
                    # 统计split使用情况
                    split_key = str(best_split)
                    split_stats[split_key] = split_stats.get(split_key, 0) + 1
                    
                    # 在原始数据中添加split策略
                    enhanced_data = prompt_data.copy()
                    enhanced_data['best_split'] = best_split
                    enhanced_data['split_score'] = float(best_score)
                    
                    out_f.write(json.dumps(enhanced_data, ensure_ascii=False) + '\n')
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理 {img_name} 时出错: {e}")
            failed_count += 1
            continue
    
    # 保存统计结果
    stats = {
        'processed_files': processed_count,
        'failed_files': failed_count,
        'split_usage_stats': split_stats
    }
    
    with open(os.path.join(output_dir, 'processing_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"失败: {failed_count} 个文件")
    print(f"结果保存在: {output_dir}")
    
    print("\nSplit策略使用统计:")
    for split, count in sorted(split_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {split}: {count} 次")
    
    return stats

def test_single_sample(cfg):
    """测试单个样本（用于调试）"""
    image_dir = cfg["data"]["images"]
    text_dir = cfg["data"]["texts"]
    mask_dir = cfg["data"]["masks"]
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    while True:
        img_name = random.choice(image_files)
        try:
            image_id = img_name.split('_')[-1].split('.')[0].zfill(12)
        except (IndexError, ValueError):
            continue
        
        text_path = os.path.join(text_dir, f"{image_id}.txt")
        mask_path = os.path.join(mask_dir, f"{image_id}.txt")
        
        if not (os.path.exists(text_path) and os.path.exists(mask_path)):
            continue
        
        img_path = os.path.join(image_dir, img_name)
        prompts = _read_json_lines(text_path)
        instances = _read_json_lines(mask_path)
        
        if not prompts or not instances:
            continue
        
        instance_map = {inst['ann_id']: inst for inst in instances}
        valid_prompts = [p for p in prompts if p['ann_id'] in instance_map]
        
        if not valid_prompts:
            continue
        
        prompt_data = random.choice(valid_prompts)
        ann_data = instance_map[prompt_data['ann_id']]
        prompt = prompt_data['sent']
        
        with Image.open(img_path) as img:
            orig_wh = img.size
        gt_mask = _bbox_to_mask(ann_data["bbox"], orig_wh=orig_wh)
        
        print(f"测试样本: {image_id}")
        best_split, best_score, results = find_best_split(img_path, prompt, gt_mask, verbose=True)
        
        return image_id, best_split, best_score

def main():
    # 加载配置
    cfg = yaml.safe_load(open("config.yaml"))
    
    print("选择运行模式:")
    print("1. 测试单个样本")
    print("2. 处理所有样本")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_single_sample(cfg)
    elif choice == "2":
        output_dir = input("请输入输出目录名称 (默认: split_results): ").strip() or "split_results"
        process_all_samples(cfg, output_dir)
    else:
        print("无效选择")

if __name__ == "__main__":
    main()