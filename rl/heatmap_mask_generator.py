import os
import json
import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
import matplotlib.pyplot as plt
from utils.clip_util import get_heatmap

class HeatmapMaskGenerator:
    def __init__(self):
        self.methods = {
            'connected_components': self.method_connected_components,
            'watershed': self.method_watershed,
            'adaptive_threshold': self.method_adaptive_threshold,
            'region_growing': self.method_region_growing
        }
    
    def method_connected_components(self, heatmap, threshold=0.5, min_area=100):
        """
        方法1：阈值化 + 连通域分析
        """
        # 1. 归一化热力图
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 2. 阈值化
        binary_mask = (heatmap_norm > threshold).astype(np.uint8)
        
        # 3. 形态学操作，填补小洞
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 4. 连通域分析
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels <= 1:
            return binary_mask, 0.0
        
        # 5. 找到最大的连通域（排除背景）
        largest_component = 0
        largest_area = 0
        
        for i in range(1, num_labels):
            area = np.sum(labels == i)
            if area > largest_area and area > min_area:
                largest_area = area
                largest_component = i
        
        # 6. 生成最终mask
        final_mask = (labels == largest_component).astype(np.uint8)
        
        # 7. 计算与原始热力图的匹配度
        score = np.sum(heatmap_norm * final_mask) / np.sum(final_mask) if np.sum(final_mask) > 0 else 0
        
        return final_mask, score
    
    def method_watershed(self, heatmap, min_distance=10, threshold_abs=0.3):
        """
        方法2：Watershed分割
        """
        # 1. 归一化
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 2. 高斯平滑
        smoothed = filters.gaussian(heatmap_norm, sigma=1.0)
        
        # 3. 找到局部最大值作为种子点
        local_maxima = peak_local_maxima(
            smoothed, 
            min_distance=min_distance, 
            threshold_abs=threshold_abs
        )
        
        if len(local_maxima) == 0:
            # 如果没有找到局部最大值，使用全局最大值
            max_pos = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            local_maxima = np.array([max_pos])
        
        # 4. 创建标记
        markers = np.zeros_like(smoothed, dtype=int)
        for i, pos in enumerate(local_maxima):
            markers[tuple(pos)] = i + 1
        
        # 5. 创建mask（高于阈值的区域）
        mask = smoothed > np.percentile(smoothed, 70)
        
        # 6. Watershed分割
        labels = watershed(-smoothed, markers, mask=mask)
        
        # 7. 选择包含最高值的区域
        max_pos = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        target_label = labels[max_pos]
        
        final_mask = (labels == target_label).astype(np.uint8)
        
        # 8. 计算得分
        score = np.sum(heatmap_norm * final_mask) / np.sum(final_mask) if np.sum(final_mask) > 0 else 0
        
        return final_mask, score
    
    def method_adaptive_threshold(self, heatmap, percentile=80):
        """
        方法3：自适应阈值
        """
        # 1. 归一化
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 2. 计算自适应阈值
        threshold = np.percentile(heatmap_norm, percentile)
        
        # 3. 生成初始mask
        binary_mask = (heatmap_norm >= threshold).astype(np.uint8)
        
        # 4. 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 5. 区域生长
        grown_mask = ndimage.binary_dilation(binary_mask, iterations=2)
        
        # 6. 保留最大连通域
        num_labels, labels = cv2.connectedComponents(grown_mask.astype(np.uint8))
        
        if num_labels <= 1:
            final_mask = grown_mask.astype(np.uint8)
        else:
            # 找到最大连通域
            largest_component = 0
            largest_area = 0
            
            for i in range(1, num_labels):
                area = np.sum(labels == i)
                if area > largest_area:
                    largest_area = area
                    largest_component = i
            
            final_mask = (labels == largest_component).astype(np.uint8)
        
        # 7. 计算得分
        score = np.sum(heatmap_norm * final_mask) / np.sum(final_mask) if np.sum(final_mask) > 0 else 0
        
        return final_mask, score
    
    def method_region_growing(self, heatmap, seed_percentile=95, growth_threshold=0.3):
        """
        方法4：区域生长
        """
        # 1. 归一化
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 2. 找到种子点（最高值区域）
        seed_threshold = np.percentile(heatmap_norm, seed_percentile)
        seed_mask = heatmap_norm >= seed_threshold
        
        # 3. 区域生长
        grown_mask = seed_mask.copy()
        
        # 迭代生长
        for _ in range(5):
            # 膨胀操作
            dilated = ndimage.binary_dilation(grown_mask)
            
            # 只保留热力图值高于阈值的区域
            grown_mask = dilated & (heatmap_norm > growth_threshold)
        
        # 4. 保留最大连通域
        num_labels, labels = cv2.connectedComponents(grown_mask.astype(np.uint8))
        
        if num_labels <= 1:
            final_mask = grown_mask.astype(np.uint8)
        else:
            # 找到包含原始种子点的连通域
            seed_positions = np.where(seed_mask)
            if len(seed_positions[0]) > 0:
                seed_pos = (seed_positions[0][0], seed_positions[1][0])
                target_label = labels[seed_pos]
                final_mask = (labels == target_label).astype(np.uint8)
            else:
                # 如果没有种子点，选择最大连通域
                largest_component = 0
                largest_area = 0
                
                for i in range(1, num_labels):
                    area = np.sum(labels == i)
                    if area > largest_area:
                        largest_area = area
                        largest_component = i
                
                final_mask = (labels == largest_component).astype(np.uint8)
        
        # 5. 计算得分
        score = np.sum(heatmap_norm * final_mask) / np.sum(final_mask) if np.sum(final_mask) > 0 else 0
        
        return final_mask, score
    
    def generate_mask(self, heatmap, method='connected_components', **kwargs):
        """
        生成mask的主函数
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        return self.methods[method](heatmap, **kwargs)
    
    def compare_methods(self, heatmap, methods=None):
        """
        比较不同方法的效果
        """
        if methods is None:
            methods = list(self.methods.keys())
        
        results = {}
        
        for method in methods:
            try:
                mask, score = self.generate_mask(heatmap, method)
                results[method] = {
                    'mask': mask,
                    'score': score,
                    'area': np.sum(mask),
                    'coverage': np.sum(mask) / (mask.shape[0] * mask.shape[1])
                }
            except Exception as e:
                print(f"Method {method} failed: {e}")
                results[method] = None
        
        return results
    
    def visualize_results(self, heatmap, results, save_path=None):
        """
        可视化结果
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 显示原始热力图
        axes[0].imshow(heatmap, cmap='hot')
        axes[0].set_title('Original Heatmap')
        axes[0].axis('off')
        
        # 显示各种方法的结果
        for i, (method, result) in enumerate(results.items()):
            if result is not None and i + 1 < len(axes):
                axes[i + 1].imshow(result['mask'], cmap='gray')
                axes[i + 1].set_title(f"{method}\nScore: {result['score']:.3f}\nArea: {result['area']}")
                axes[i + 1].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(results) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

def test_single_sample(img_path, prompt, output_dir='mask_results'):
    """
    测试单个样本
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取热力图
    print(f"Processing: {img_path}")
    print(f"Prompt: {prompt}")
    
    heatmap = get_heatmap(img_path, prompt)
    
    # 创建mask生成器
    generator = HeatmapMaskGenerator()
    
    # 比较不同方法
    results = generator.compare_methods(heatmap)
    
    # 显示结果
    print("\n=== Results ===")
    best_method = None
    best_score = 0
    
    for method, result in results.items():
        if result is not None:
            print(f"{method:20s}: Score={result['score']:.4f}, Area={result['area']:4d}, Coverage={result['coverage']:.3f}")
            if result['score'] > best_score:
                best_score = result['score']
                best_method = method
        else:
            print(f"{method:20s}: Failed")
    
    print(f"\nBest method: {best_method} (Score: {best_score:.4f})")
    
    # 可视化结果
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"{img_name}_comparison.png")
    generator.visualize_results(heatmap, results, save_path)
    
    return results, best_method

def process_all_samples(data_dir, output_dir='mask_results'):
    """
    处理所有样本
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据文件
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    generator = HeatmapMaskGenerator()
    all_results = []
    method_stats = {method: {'count': 0, 'total_score': 0} for method in generator.methods.keys()}
    
    for txt_file in txt_files:
        txt_path = os.path.join(data_dir, txt_file)
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        img_path = data['image']
                        prompt = data['sent']
                        
                        # 检查图片文件是否存在
                        if not os.path.exists(img_path):
                            print(f"Image not found: {img_path}")
                            continue
                        
                        # 获取热力图
                        heatmap = get_heatmap(img_path, prompt)
                        
                        # 比较不同方法
                        results = generator.compare_methods(heatmap)
                        
                        # 找到最佳方法
                        best_method = None
                        best_score = 0
                        
                        for method, result in results.items():
                            if result is not None and result['score'] > best_score:
                                best_score = result['score']
                                best_method = method
                        
                        # 更新统计
                        if best_method:
                            method_stats[best_method]['count'] += 1
                            method_stats[best_method]['total_score'] += best_score
                        
                        # 保存增强数据
                        enhanced_data = data.copy()
                        enhanced_data['best_mask_method'] = best_method
                        enhanced_data['mask_score'] = float(best_score)
                        
                        # 保存到新文件
                        output_file = os.path.join(output_dir, f"enhanced_{txt_file}")
                        with open(output_file, 'a', encoding='utf-8') as out_f:
                            out_f.write(json.dumps(enhanced_data, ensure_ascii=False) + '\n')
                        
                        all_results.append({
                            'file': txt_file,
                            'line': line_num,
                            'image': img_path,
                            'prompt': prompt,
                            'best_method': best_method,
                            'score': best_score
                        })
                        
                        print(f"Processed: {img_path} -> {best_method} (Score: {best_score:.4f})")
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in {txt_file} line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing {txt_file} line {line_num}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error reading file {txt_file}: {e}")
            continue
    
    # 保存统计结果
    stats = {
        'total_processed': len(all_results),
        'method_stats': {}
    }
    
    for method, stat in method_stats.items():
        if stat['count'] > 0:
            stats['method_stats'][method] = {
                'count': stat['count'],
                'avg_score': stat['total_score'] / stat['count'],
                'percentage': stat['count'] / len(all_results) * 100
            }
    
    stats_file = os.path.join(output_dir, 'mask_generation_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total processed: {len(all_results)}")
    print(f"Statistics saved to: {stats_file}")
    
    for method, stat in stats['method_stats'].items():
        print(f"{method:20s}: {stat['count']:3d} samples ({stat['percentage']:5.1f}%), avg_score={stat['avg_score']:.4f}")
    
    return all_results, stats

if __name__ == "__main__":
    print("Heatmap Mask Generator")
    print("=" * 50)
    
    mode = input("选择运行模式:\n1. 测试单个样本\n2. 处理所有样本\n请输入选择 (1/2): ")
    
    if mode == '1':
        # 测试单个样本
        img_path = input("请输入图片路径: ")
        prompt = input("请输入文本描述: ")
        
        if os.path.exists(img_path):
            results, best_method = test_single_sample(img_path, prompt)
        else:
            print(f"图片文件不存在: {img_path}")
    
    elif mode == '2':
        # 处理所有样本
        data_dir = input("请输入数据目录路径: ")
        
        if os.path.exists(data_dir):
            results, stats = process_all_samples(data_dir)
        else:
            print(f"数据目录不存在: {data_dir}")
    
    else:
        print("无效的选择")