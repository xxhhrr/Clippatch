import os
import json
import random
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path

class SimpleMaskVisualizer:
    def __init__(self, config_path="config.yaml", output_dir="visualization_output"):
        # 从config.yaml加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 设置路径（相对于config.yaml文件的位置）
        config_dir = Path(config_path).parent
        self.images_dir = config_dir / config['data']['images']
        self.masks_dir = config_dir / config['data']['masks']  # 改为masks_dir
        self.texts_dir = config_dir / config['data']['texts']
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色定义：预测框用红色，真实框用绿色
        self.pred_color = (255, 0, 0)  # 红色 - 预测框
        self.gt_color = (0, 255, 0)    # 绿色 - 真实框
    
    def _read_json_lines(self, path):
        """读取JSON行文件，静默处理错误"""
        results = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    try:
                        data = json.loads(line)
                        # 验证必需字段 - 静默跳过缺少字段的行
                        if 'sent' not in data or 'ann_id' not in data:
                            continue  # 静默跳过，不打印错误
                        results.append(data)
                    except json.JSONDecodeError:
                        continue  # 静默跳过JSON解析错误
        except Exception:
            return []  # 静默返回空列表
        
        return results
    
    def load_ground_truth_mask(self, image_id, ann_id):
        """从masks目录加载真实mask（静默处理错误）"""
        # masks文件是.txt格式的JSON Lines文件
        mask_file = self.masks_dir / f"{image_id}.txt"
        if not mask_file.exists():
            return None  # 静默返回None
            
        try:
            # 读取JSON Lines格式的mask文件
            mask_data = self._read_json_lines(mask_file)
            
            # 查找对应ann_id的mask数据
            target_mask = None
            for mask_info in mask_data:
                if mask_info.get('ann_id') == ann_id:
                    target_mask = mask_info
                    break
            
            if target_mask is None:
                return None  # 静默返回None，不打印错误
            
            # 根据bbox信息创建mask
            if 'bbox' in target_mask:
                bbox = target_mask['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # 获取原始图像尺寸来创建正确大小的mask
                img_path = self.images_dir / f"COCO_train2014_{image_id}.jpg"
                if not img_path.exists():
                    img_path = self.images_dir / f"{image_id}.jpg"
                
                if img_path.exists():
                    original_image = cv2.imread(str(img_path))
                    if original_image is not None:
                        original_height, original_width = original_image.shape[:2]
                        
                        # 创建与原始图像同尺寸的mask
                        mask = np.zeros((original_height, original_width), dtype=np.uint8)
                        
                        # 将bbox区域标记为255
                        x1, y1 = int(x), int(y)
                        x2, y2 = int(x + w), int(y + h)
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, original_width))
                        y1 = max(0, min(y1, original_height))
                        x2 = max(0, min(x2, original_width))
                        y2 = max(0, min(y2, original_height))
                        
                        mask[y1:y2, x1:x2] = 255
                        return mask
                
                # 如果无法获取原始图像尺寸，使用默认尺寸
                mask = np.zeros((640, 640), dtype=np.uint8)  # 默认尺寸
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                mask[y1:y2, x1:x2] = 255
                return mask
                
        except Exception:
            return None  # 静默返回None
            
        return None
    
    def mask_to_bbox(self, mask):
        """将mask转换为边界框"""
        if mask is None or np.sum(mask) == 0:
            return None
            
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 返回 [x1, y1, x2, y2] 格式
        return [x_min, y_min, x_max, y_max]
    
    def generate_predicted_mask(self, img_path, prompt):
        """生成预测mask（修正版）"""
        try:
            from utils.clip_util import get_heatmap
            
            # 获取原始图像尺寸
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                return None
            original_height, original_width = original_image.shape[:2]
            
            # 获取224x224的热力图
            heatmap = get_heatmap(str(img_path), prompt)
            
            # 将热力图缩放到原始图像尺寸
            heatmap_resized = cv2.resize(heatmap, (original_width, original_height), 
                                       interpolation=cv2.INTER_CUBIC)
            
            # 简单阈值化生成mask
            threshold = np.percentile(heatmap_resized, 80)  # 取前20%的高值区域
            pred_mask = (heatmap_resized > threshold).astype(np.uint8) * 255
            
            return pred_mask
        except Exception as e:
            print(f"生成预测mask失败: {e}")
            return None
    
    def draw_boxes_on_image(self, image, pred_bbox, gt_bbox, prompt):
        """在图像上绘制预测框和真实框"""
        # 转换为PIL图像以便绘制
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        draw = ImageDraw.Draw(image)
        
        # 绘制真实框（绿色，较粗的线）
        if gt_bbox is not None:
            x1, y1, x2, y2 = gt_bbox
            draw.rectangle([x1, y1, x2, y2], outline=self.gt_color, width=3)
            draw.text((x1, y1-20), "Ground Truth", fill=self.gt_color)
        
        # 绘制预测框（红色，较细的线）
        if pred_bbox is not None:
            x1, y1, x2, y2 = pred_bbox
            draw.rectangle([x1, y1, x2, y2], outline=self.pred_color, width=2)
            draw.text((x1, y2+5), "Predicted", fill=self.pred_color)
        
        # 添加文本提示
        draw.text((10, 10), f"Prompt: {prompt[:50]}...", fill=(255, 255, 255))
        
        return image
    
    def collect_all_samples(self):
        """从texts目录收集所有样本"""
        all_samples = []
        
        # 遍历texts目录下的所有txt文件
        for txt_file in self.texts_dir.glob("*.txt"):
            try:
                prompts = self._read_json_lines(txt_file)
                for prompt_data in prompts:
                    all_samples.append(prompt_data)
            except Exception as e:
                print(f"读取文件失败 {txt_file}: {e}")
                continue
        
        return all_samples
    
    def collect_samples_from_images(self):
        """从images目录收集样本"""
        # 获取所有图片文件
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))]
        print(f"找到 {len(image_files)} 个图片文件")
        
        all_samples = []
        
        for i, img_name in enumerate(image_files):
            if i % 10000 == 0:
                print(f"处理进度: {i}/{len(image_files)}")
                
            # 提取图片ID（和simple_split_selector.py一样的方法）
            image_id = img_name.split('_')[-1].split('.')[0].zfill(12)
            
            # 直接构建对应的text和mask文件路径
            text_path = self.texts_dir / f"{image_id}.txt"
            
            mask_path = self.masks_dir / f"{image_id}.txt"
            
            # 检查文件是否存在
            if not (text_path.exists() and mask_path.exists()):
                continue
                
            # 读取text文件中的所有样本
            try:
                prompts = self._read_json_lines(text_path)
                img_path = self.images_dir / img_name
                
                for prompt_data in prompts:
                    # 添加缺失的字段
                    prompt_data['image_id'] = image_id  # 添加image_id
                    prompt_data['image_path'] = str(img_path)
                    all_samples.append(prompt_data)
                
            except Exception as e:
                print(f"处理样本失败 {text_path}: {e}")
                continue
        
        print(f"总共收集到 {len(all_samples)} 个样本")
        return all_samples
    
    def generate_random_visualizations(self, num_samples=1000):
        """生成随机选择的可视化结果（简化版）"""
        print(f"开始收集样本（目标: {num_samples} 个）...")
        
        # 使用简化的样本收集方法
        all_samples = self.collect_samples_from_images()
        
        if len(all_samples) == 0:
            print("没有找到任何有效样本！")
            return []
        
        print(f"总共找到 {len(all_samples)} 个有效样本")
        
        # 随机选择样本
        selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        print(f"随机选择了 {len(selected_samples)} 个样本进行可视化")
        
        results = []
        success_count = 0
        
        for i, sample in enumerate(selected_samples):
            if i % 100 == 0:
                print(f"可视化进度: {i}/{len(selected_samples)} ({i/len(selected_samples)*100:.1f}%)")
            
            result = self.process_single_sample(sample)
            if result is not None:
                # 保存可视化图像
                output_path = self.output_dir / f"sample_{result['image_id']}_{result['ann_id']}.jpg"
                result['result_image'].save(output_path)
                
                results.append({
                    'image_id': result['image_id'],
                    'ann_id': result['ann_id'],
                    'prompt': result['prompt'],
                    'output_path': str(output_path),
                    'has_pred_bbox': result['pred_bbox'] is not None,
                    'has_gt_bbox': result['gt_bbox'] is not None
                })
                
                success_count += 1
        
        # 生成统计报告
        self.generate_summary_report(results)
        
        print(f"\n可视化完成！")
        print(f"成功处理: {success_count}/{len(selected_samples)} 个样本")
        print(f"输出目录: {self.output_dir}")
        
        return results
    
    def process_single_sample(self, sample_data):
        """处理单个样本（静默处理错误）"""
        try:
            image_id = sample_data['image_id']
            ann_id = sample_data['ann_id']
            prompt = sample_data['sent']
            
            # 使用预先确定的图像路径
            if 'image_path' in sample_data:
                img_path = sample_data['image_path']
            else:
                # 回退到原来的逻辑
                img_path = self.images_dir / f"COCO_train2014_{image_id}.jpg"
                if not img_path.exists():
                    img_path = self.images_dir / f"{image_id}.jpg"
                    if not img_path.exists():
                        return None  # 静默返回None
            
            # 加载图像
            image = cv2.imread(str(img_path))
            if image is None:
                return None  # 静默返回None
            
            # 加载真实mask - 如果找不到对应ann_id，静默跳过
            gt_mask = self.load_ground_truth_mask(image_id, ann_id)
            if gt_mask is None:
                return None  # 静默跳过没有对应ann_id的样本
            
            # 生成预测mask
            pred_mask = self.generate_predicted_mask(str(img_path), prompt)
            pred_bbox = self.mask_to_bbox(pred_mask) if pred_mask is not None else None
            gt_bbox = self.mask_to_bbox(gt_mask)
            
            # 在图像上绘制框
            result_image = self.draw_boxes_on_image(image, pred_bbox, gt_bbox, prompt)
            
            return {
                'image_id': image_id,
                'ann_id': ann_id,
                'prompt': prompt,
                'result_image': result_image,
                'pred_bbox': pred_bbox,
                'gt_bbox': gt_bbox
            }
            
        except Exception:
            return None  # 静默返回None
    
    def generate_summary_report(self, results):
        """生成汇总报告"""
        report_path = self.output_dir / "visualization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 可视化结果报告 ===\n\n")
            f.write(f"总样本数: {len(results)}\n")
            f.write(f"有预测框的样本: {sum(1 for r in results if r['has_pred_bbox'])}\n")
            f.write(f"有真实框的样本: {sum(1 for r in results if r['has_gt_bbox'])}\n\n")
            
            f.write("样本详情:\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. 图像ID: {result['image_id']}, "
                       f"标注ID: {result['ann_id']}, "
                       f"提示: {result['prompt'][:30]}...\n")
                f.write(f"   输出文件: {result['output_path']}\n")
                f.write(f"   预测框: {'✓' if result['has_pred_bbox'] else '✗'}, "
                       f"真实框: {'✓' if result['has_gt_bbox'] else '✗'}\n\n")

# 使用示例
if __name__ == "__main__":
    # 创建可视化器（自动从config.yaml加载所有路径配置）
    visualizer = SimpleMaskVisualizer(
        config_path="config.yaml",
        output_dir="mask_visualization_output"
    )
    
    # 生成1000个随机样本的可视化（直接从texts目录读取所有数据）
    results = visualizer.generate_random_visualizations(num_samples=1000)