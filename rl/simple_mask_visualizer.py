import os
import json
import random
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

class SimpleMaskVisualizer:
    def __init__(self, config_path="config.yaml", output_dir="visualization_output"):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 设置路径
        config_dir = Path(config_path).parent
        self.images_dir = config_dir / config['data']['images']
        self.masks_dir = config_dir / config['data']['masks']
        self.texts_dir = config_dir / config['data']['texts']
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色定义
        self.pred_color = (255, 0, 0)  # 红色 - 预测框
        self.gt_color = (0, 255, 0)    # 绿色 - 真实框
    
    def _read_json_lines(self, path):
        """读取JSON Lines文件"""
        results = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return []
        return results
    
    def load_ground_truth_mask(self, image_id, ann_id):
        """加载真实bbox"""
        mask_file = self.masks_dir / f"{image_id}.txt"
        if not mask_file.exists():
            return None
            
        try:
            mask_data = self._read_json_lines(mask_file)
            for mask_info in mask_data:
                if mask_info.get('ann_id') == ann_id and 'bbox' in mask_info:
                    return mask_info['bbox']
        except Exception:
            pass
        return None
    
    def _clean_heatmap_morphology(self, heatmap, threshold=0.1):
        """使用形态学操作清理热图噪声"""
        # 阈值化
        binary = (heatmap > threshold).astype(np.uint8)
        
        # 形态学开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 应用到原热图
        cleaned_heatmap = heatmap * cleaned_binary
        
        # 重新归一化
        if cleaned_heatmap.max() > 0:
            cleaned_heatmap = (cleaned_heatmap - cleaned_heatmap.min()) / (cleaned_heatmap.max() + 1e-6)
        
        return cleaned_heatmap
    
    def generate_predicted_mask_with_heatmap(self, img_path, prompt):
        """生成预测mask并返回热图"""
        try:
            from utils.clip_util import get_heatmap
            
            # 获取原始图像尺寸
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                return None, None
            original_height, original_width = original_image.shape[:2]
            
            # 获取224x224的热力图
            heatmap = get_heatmap(str(img_path), prompt)
            
            # 放大到与原图同尺寸并归一化
            heatmap_resized = cv2.resize(heatmap, (original_width, original_height), 
                                       interpolation=cv2.INTER_CUBIC)
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            
            # 简化的阈值处理
            high_threshold = np.percentile(heatmap_resized, 90)
            binary_mask = (heatmap_resized > high_threshold).astype(np.uint8)
            
            # 形态学处理
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # 找到最大连通域作为预测结果
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                final_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                final_mask[y:y+h, x:x+w] = 255
            else:
                final_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            
            return final_mask, heatmap_resized
            
        except Exception as e:
            print(f"生成预测mask失败: {e}")
            return None, None
    
    def mask_to_bbox(self, mask):
        """将mask转换为bbox"""
        if mask is None:
            return None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, x+w, y+h]
    
    def calculate_bbox_iou(self, bbox1, bbox2):
        """计算两个bbox的IoU"""
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        # 转换为[x1, y1, x2, y2]格式
        if len(bbox1) == 4 and len(bbox2) == 4:
            # 如果是[x, y, w, h]格式，转换为[x1, y1, x2, y2]
            if bbox1[2] < bbox1[0] or bbox1[3] < bbox1[1]:  # 已经是[x1,y1,x2,y2]
                box1 = bbox1
            else:  # 是[x,y,w,h]格式
                box1 = [bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]]
            
            if bbox2[2] < bbox2[0] or bbox2[3] < bbox2[1]:  # 已经是[x1,y1,x2,y2]
                box2 = bbox2
            else:  # 是[x,y,w,h]格式
                box2 = [bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]]
        else:
            return 0.0
        
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_debug_visualization(self, image, pred_bbox, gt_bbox, prompt, heatmap):
        """创建调试可视化图像（简化版）"""
        try:
            # 转换图像格式
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            draw = ImageDraw.Draw(pil_image)
            
            # 绘制预测框（红色）
            if pred_bbox is not None:
                x1, y1, x2, y2 = pred_bbox
                draw.rectangle([x1, y1, x2, y2], outline=self.pred_color, width=3)
                draw.text((x1, y1-25), "Pred", fill=self.pred_color)
            
            # 绘制真实框（绿色）
            if gt_bbox is not None:
                x1, y1, x2, y2 = gt_bbox
                draw.rectangle([x1, y1, x2, y2], outline=self.gt_color, width=3)
                draw.text((x1, y1-50), "GT", fill=self.gt_color)
            
            # 添加prompt文本
            if prompt:
                draw.text((10, 10), f"Prompt: {prompt}", fill=(255, 255, 255))
            
            # 转换回numpy数组
            result_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # === 关键改进：使用清理后的热图 ===
            from utils.clip_util import get_heatmap
            
            # 获取原始热图
            if hasattr(self, 'current_img_path'):
                original_heatmap = get_heatmap(str(self.current_img_path), prompt)
            else:
                original_heatmap = heatmap
            
            # 清理热图噪声
            cleaned_heatmap = self._clean_heatmap_morphology(original_heatmap, threshold=0.1)
            
            # 转换为可视化格式
            heatmap_colored = cv2.applyColorMap((cleaned_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # 调整尺寸
            h, w = result_with_boxes.shape[:2]
            heatmap_colored = cv2.resize(heatmap_colored, (w, h))
            
            # 创建组合图像
            combined = np.hstack([result_with_boxes, heatmap_colored])
            
            return combined
            
        except Exception as e:
            print(f"创建可视化失败: {e}")
            return None
    
    def process_single_sample(self, sample_data):
        """处理单个样本"""
        try:
            image_id = sample_data['image_id']
            ann_id = sample_data['ann_id']
            prompt = sample_data.get('prompt', sample_data.get('text', ''))
            
            # 保存当前图像路径供可视化使用
            img_path = self.images_dir / f"{image_id}.jpg"
            self.current_img_path = img_path
            
            if not img_path.exists():
                return None
            
            # 加载图像
            image = cv2.imread(str(img_path))
            if image is None:
                return None
            
            # 生成预测
            pred_mask, heatmap = self.generate_predicted_mask_with_heatmap(img_path, prompt)
            if pred_mask is None:
                return None
            
            # 获取预测bbox
            pred_bbox = self.mask_to_bbox(pred_mask)
            
            # 加载真实bbox
            gt_bbox = self.load_ground_truth_mask(image_id, ann_id)
            
            # 计算IoU
            iou = self.calculate_bbox_iou(pred_bbox, gt_bbox)
            
            # 创建可视化
            debug_image = self.create_debug_visualization(image, pred_bbox, gt_bbox, prompt, heatmap)
            
            return {
                'image_id': image_id,
                'ann_id': ann_id,
                'prompt': prompt,
                'pred_bbox': pred_bbox,
                'gt_bbox': gt_bbox,
                'iou': iou,
                'debug_image': debug_image
            }
            
        except Exception as e:
            print(f"处理样本失败: {e}")
            return None
    
    def collect_all_samples(self):
        """收集所有样本"""
        samples = []
        
        # 从texts目录收集所有样本
        for text_file in self.texts_dir.glob("*.txt"):
            text_data = self._read_json_lines(text_file)
            samples.extend(text_data)
        
        return samples
    
    def generate_random_visualizations(self, num_samples=100):
        """生成随机可视化"""
        all_samples = self.collect_all_samples()
        if not all_samples:
            print("没有找到样本数据")
            return []
        
        # 随机选择样本
        selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        results = []
        for i, sample in enumerate(selected_samples):
            print(f"处理样本 {i+1}/{len(selected_samples)}")
            
            result = self.process_single_sample(sample)
            if result and result['debug_image'] is not None:
                # 保存可视化结果
                output_path = self.output_dir / f"debug_{result['image_id']}_{result['ann_id']}.jpg"
                cv2.imwrite(str(output_path), result['debug_image'])
                
                results.append({
                    'image_id': result['image_id'],
                    'ann_id': result['ann_id'],
                    'iou': result['iou'],
                    'output_path': str(output_path)
                })
        
        print(f"完成处理，生成了 {len(results)} 个可视化结果")
        return results

if __name__ == "__main__":
    visualizer = SimpleMaskVisualizer(
        config_path="config.yaml",
        output_dir="mask_visualization_output"
    )
    
    results = visualizer.generate_random_visualizations(num_samples=100)