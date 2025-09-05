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
        self.instance_dir = config_dir / config['data']['masks']
        self.texts_dir = config_dir / config['data']['texts']
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色定义：预测框用红色，真实框用绿色
        self.pred_color = (255, 0, 0)  # 红色 - 预测框
        self.gt_color = (0, 255, 0)    # 绿色 - 真实框
    
    def _read_json_lines(self, path):
        """读取JSON行文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(l.strip()) for l in f if l.strip()]
        
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
        """生成预测mask"""
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
        """从images目录收集样本（正确的流程）"""
        all_samples = []
        
        # 获取所有图片文件
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        print(f"找到 {len(image_files)} 个图片文件")
        
        for img_file in image_files:
            # 从文件名提取image_id
            image_id = self._extract_image_id(img_file.name)
            if not image_id:
                continue
                
            # 查找对应的text文件
            text_samples = self._find_text_samples_for_image(image_id)
            
            # 检查是否有对应的instance文件
            instance_file = self.instance_dir / f"{image_id}.png"
            if not instance_file.exists():
                continue
                
            # 添加所有有效的样本
            for sample in text_samples:
                sample['image_path'] = str(img_file)
                all_samples.append(sample)
        
        return all_samples
    
    def _extract_image_id(self, filename):
        """从文件名提取image_id"""
        # 处理 COCO_train2014_000000123456.jpg 格式
        if filename.startswith('COCO_train2014_'):
            return filename.replace('COCO_train2014_', '').replace('.jpg', '').replace('.png', '')
        # 处理 123456.jpg 格式
        elif filename.replace('.jpg', '').replace('.png', '').isdigit():
            return filename.replace('.jpg', '').replace('.png', '')
        else:
            return None
    
    def _find_text_samples_for_image(self, image_id):
        """根据image_id查找对应的text样本"""
        samples = []
        
        # 遍历所有text文件
        for txt_file in self.texts_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            # 检查是否匹配当前image_id
                            if str(data.get('image_id', '')) == str(image_id):
                                samples.append(data)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"读取文件失败 {txt_file}: {e}")
                continue
        
        return samples
    
    def generate_random_visualizations(self, num_samples=1000):
        """生成随机选择的可视化结果（修正版）"""
        print(f"开始从images目录收集样本...")
        
        # 使用正确的流程收集样本
        all_samples = self.collect_samples_from_images()
        print(f"总共找到 {len(all_samples)} 个有效样本")
        
        if len(all_samples) == 0:
            print("没有找到任何有效样本！")
            print("请检查：")
            print(f"1. images目录: {self.images_dir}")
            print(f"2. texts目录: {self.texts_dir}")
            print(f"3. instance目录: {self.instance_dir}")
            return []
        
        # 随机选择样本
        selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        print(f"随机选择了 {len(selected_samples)} 个样本进行可视化")
        
        results = []
        success_count = 0
        
        for i, sample in enumerate(selected_samples):
            print(f"处理进度: {i+1}/{len(selected_samples)}")
            
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
            else:
                print(f"处理样本失败: image_id={sample.get('image_id')}, ann_id={sample.get('ann_id')}")
        
        # 生成统计报告
        self.generate_summary_report(results)
        
        print(f"\n可视化完成！")
        print(f"成功处理: {success_count}/{len(selected_samples)} 个样本")
        print(f"输出目录: {self.output_dir}")
        
        return results
    
    def process_single_sample(self, sample_data):
        """处理单个样本（修正版）"""
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
                        print(f"图像文件不存在: {image_id}")
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