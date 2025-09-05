import os
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path

class SimpleMaskVisualizer:
    def __init__(self, data_dir, instance_dir, output_dir="visualization_output"):
        self.data_dir = Path(data_dir)
        self.instance_dir = Path(instance_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色定义：预测框用红色，真实框用绿色
        self.pred_color = (255, 0, 0)  # 红色 - 预测框
        self.gt_color = (0, 255, 0)    # 绿色 - 真实框
        
    def load_ground_truth_mask(self, image_id, ann_id):
        """从instance目录加载真实mask"""
        mask_file = self.instance_dir / f"{image_id}.png"
        if not mask_file.exists():
            return None
            
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        # 找到对应ann_id的mask区域
        gt_mask = (mask == ann_id).astype(np.uint8) * 255
        return gt_mask
    
    def mask_to_bbox(self, mask):
        """将mask转换为边界框"""
        if mask is None or np.sum(mask) == 0:
            return None
            
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        return [x_min, y_min, x_max, y_max]
    
    def generate_predicted_mask(self, img_path, prompt):
        """生成预测mask（这里使用您的heatmap_mask_generator逻辑）"""
        # 这里应该调用您的热力图生成逻辑
        # 暂时返回一个示例mask，您需要替换为实际的热力图生成代码
        try:
            from utils.clip_util import get_heatmap
            
            # 获取热力图
            heatmap = get_heatmap(img_path, prompt)
            
            # 简单阈值化生成mask
            threshold = np.percentile(heatmap, 80)  # 取前20%的高值区域
            pred_mask = (heatmap > threshold).astype(np.uint8) * 255
            
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
    
    def process_single_sample(self, data_line):
        """处理单个样本"""
        try:
            # 解析数据
            data = json.loads(data_line.strip())
            image_id = data['image_id']
            ann_id = data['ann_id']
            prompt = data['sent']
            
            # 构建图像路径
            img_path = self.data_dir / "images" / f"{image_id}.jpg"
            if not img_path.exists():
                print(f"图像文件不存在: {img_path}")
                return None
            
            # 加载图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"无法加载图像: {img_path}")
                return None
            
            # 生成预测mask
            pred_mask = self.generate_predicted_mask(str(img_path), prompt)
            pred_bbox = self.mask_to_bbox(pred_mask) if pred_mask is not None else None
            
            # 加载真实mask
            gt_mask = self.load_ground_truth_mask(image_id, ann_id)
            gt_bbox = self.mask_to_bbox(gt_mask) if gt_mask is not None else None
            
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
            
        except Exception as e:
            print(f"处理样本失败: {e}")
            return None
    
    def generate_random_visualizations(self, data_file, num_samples=1000):
        """生成随机选择的可视化结果"""
        print(f"开始生成 {num_samples} 个随机样本的可视化...")
        
        # 读取所有数据行
        with open(data_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # 随机选择样本
        selected_lines = random.sample(all_lines, min(num_samples, len(all_lines)))
        
        results = []
        success_count = 0
        
        for i, line in enumerate(selected_lines):
            print(f"处理进度: {i+1}/{len(selected_lines)}")
            
            result = self.process_single_sample(line)
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
        print(f"成功处理: {success_count}/{len(selected_lines)} 个样本")
        print(f"输出目录: {self.output_dir}")
        
        return results
    
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
    # 配置路径
    data_dir = "path/to/your/data"  # 包含images文件夹的数据目录
    instance_dir = "path/to/instance"  # instance目录路径
    data_file = "path/to/your/data.txt"  # 数据文件路径
    
    # 创建可视化器
    visualizer = SimpleMaskVisualizer(
        data_dir=data_dir,
        instance_dir=instance_dir,
        output_dir="mask_visualization_output"
    )
    
    # 生成1000个随机样本的可视化
    results = visualizer.generate_random_visualizations(data_file, num_samples=1000)