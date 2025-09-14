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
        """通用的读取JSON Lines文件方法"""
        results = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError:
                        continue  # 静默跳过JSON解析错误
        except Exception:
            return []  # 静默返回空列表
        
        return results
    
    def load_ground_truth_mask(self, image_id, ann_id):
        """从masks目录加载真实bbox（直接使用bbox坐标）"""
        # masks文件是.txt格式的JSON Lines文件
        mask_file = self.masks_dir / f"{image_id}.txt"
        print(f"查找mask文件: {mask_file}")
        
        if not mask_file.exists():
            print(f"Mask文件不存在: {mask_file}")
            return None
            
        try:
            # 直接读取所有JSON行
            mask_data = self._read_json_lines(mask_file)
            print(f"读取到 {len(mask_data)} 条mask数据")
            
            if len(mask_data) == 0:
                print(f"Mask文件为空或解析失败: {mask_file}")
                return None
            
            # 打印所有可用的ann_id
            available_ann_ids = [item.get('ann_id') for item in mask_data]
            print(f"文件中可用的ann_id: {available_ann_ids}")
            print(f"正在查找的ann_id: {ann_id} (类型: {type(ann_id)})")
            
            # 查找对应ann_id的mask数据
            target_mask = None
            for mask_info in mask_data:
                # 在这里验证字段是否存在
                if 'ann_id' not in mask_info:
                    continue
                current_ann_id = mask_info.get('ann_id')
                if current_ann_id == ann_id:
                    target_mask = mask_info
                    print(f"找到匹配的ann_id: {ann_id}")
                    break
            
            if target_mask is None:
                print(f"未找到ann_id {ann_id}的mask数据")
                return None
            
            # 直接返回bbox坐标，不需要创建mask
            if 'bbox' in target_mask:
                bbox = target_mask['bbox']  # [x, y, width, height]
                print(f"找到bbox: {bbox}")
                return bbox  # 直接返回bbox
            else:
                print(f"target_mask中没有bbox字段: {target_mask.keys()}")
                return None
                
        except Exception as e:
            print(f"加载mask时发生异常: {e}")
            return None
            
        return None
    
    def mask_to_bbox(self, mask):
        """将mask转换为边界框（增强调试）"""
        if mask is None:
            return None
            
        # 检查mask是否有效
        if np.sum(mask) == 0:
            print(f"警告：mask全为0，无法生成bbox")
            return None
            
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            print(f"警告：mask中没有找到非零像素")
            return None
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 确保bbox有效
        if x_min >= x_max or y_min >= y_max:
            print(f"警告：无效的bbox坐标 - x_min:{x_min}, x_max:{x_max}, y_min:{y_min}, y_max:{y_max}")
            return None
        
        # 返回 [x1, y1, x2, y2] 格式
        bbox = [x_min, y_min, x_max, y_max]
        print(f"生成bbox: {bbox}, mask形状: {mask.shape}, 非零像素数: {np.sum(mask > 0)}")
        return bbox
    
    def generate_predicted_mask_with_heatmap(self, img_path, prompt):
        """生成预测mask并返回热图用于调试（滞后阈值+NMS改进版）"""
        try:
            from utils.clip_util import get_heatmap
            
            # 获取原始图像尺寸
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                return None, None
            original_height, original_width = original_image.shape[:2]
            
            # 获取224x224的热力图
            heatmap = get_heatmap(str(img_path), prompt)
            
            # 1. 双三次插值放大到与原图同尺寸并做0-1归一化
            heatmap_resized = cv2.resize(heatmap, (original_width, original_height), 
                                       interpolation=cv2.INTER_CUBIC)
            # 确保0-1归一化
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            
            # 2. 滞后阈值（类似Canny的强弱连通思想）
            high_threshold = np.percentile(heatmap_resized, 92)  # 强响应种子
            low_threshold = np.percentile(heatmap_resized, 78)   # 弱响应区域
            
            # 获取强响应种子
            strong_mask = (heatmap_resized > high_threshold).astype(np.uint8)
            # 获取弱响应区域
            weak_mask = (heatmap_resized > low_threshold).astype(np.uint8)
            
            # 只保留与强响应连通的弱响应区域
            hysteresis_mask = self._hysteresis_threshold(strong_mask, weak_mask)
            
            # 3. 轻量的形态学开闭运算
            kernel = np.ones((3, 3), np.uint8)
            # 开运算：去小噪
            hysteresis_mask = cv2.morphologyEx(hysteresis_mask, cv2.MORPH_OPEN, kernel)
            # 闭运算：补小洞
            hysteresis_mask = cv2.morphologyEx(hysteresis_mask, cv2.MORPH_CLOSE, kernel)
            
            # 4. 连通域分析，生成候选框
            candidates = self._generate_candidates(hysteresis_mask, heatmap_resized)
            
            # 5. 候选框评分（平均热度 × 面积平方根）
            scored_candidates = self._score_candidates(candidates, heatmap_resized)
            
            # 6. 最小面积阈值过滤
            min_area = original_width * original_height * 0.001  # 最小面积为图像的0.1%
            filtered_candidates = [c for c in scored_candidates if c['area'] >= min_area]
            
            # 7. 非极大值抑制（NMS）
            nms_candidates = self._non_max_suppression(filtered_candidates, iou_threshold=0.5)
            
            # 8. 选择得分最高的候选作为最终预测
            if nms_candidates:
                best_candidate = max(nms_candidates, key=lambda x: x['score'])
                final_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                x1, y1, x2, y2 = best_candidate['bbox']
                final_mask[y1:y2, x1:x2] = 255
            else:
                # 如果没有候选，返回空mask
                final_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            
            return final_mask, heatmap_resized
            
        except Exception as e:
            print(f"生成预测mask失败: {e}")
            return None, None
    
    def _hysteresis_threshold(self, strong_mask, weak_mask):
        """滞后阈值：只保留与强响应连通的弱响应区域"""
        # 使用强响应作为种子，在弱响应区域中进行连通域扩展
        result = strong_mask.copy()
        
        # 找到强响应的连通域
        num_labels, labels = cv2.connectedComponents(strong_mask)
        
        for label in range(1, num_labels):
            # 对每个强响应连通域，在弱响应中寻找连通的区域
            seed_mask = (labels == label).astype(np.uint8)
            
            # 使用形态学膨胀来扩展到连通的弱响应区域
            kernel = np.ones((3, 3), np.uint8)
            expanded = seed_mask.copy()
            
            for _ in range(10):  # 最多扩展10次
                prev_expanded = expanded.copy()
                expanded = cv2.dilate(expanded, kernel, iterations=1)
                expanded = expanded & weak_mask  # 只在弱响应区域扩展
                
                if np.array_equal(expanded, prev_expanded):
                    break  # 没有新的扩展，停止
            
            result = result | expanded
        
        return result
    
    def _generate_candidates(self, binary_mask, heatmap):
        """从二值图生成候选框"""
        candidates = []
        
        # 连通域分析
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        for i in range(1, num_labels):
            # 获取当前连通域
            component_mask = (labels == i).astype(np.uint8)
            
            # 计算边界框
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                bbox = [x, y, x + w, y + h]
                area = w * h
                
                # 计算该区域内的平均热度
                region_heatmap = heatmap[y:y+h, x:x+w]
                region_mask = component_mask[y:y+h, x:x+w]
                avg_heat = np.mean(region_heatmap[region_mask > 0]) if np.sum(region_mask) > 0 else 0
                
                candidates.append({
                    'bbox': bbox,
                    'area': area,
                    'avg_heat': avg_heat,
                    'component_mask': component_mask
                })
        
        return candidates
    
    def _score_candidates(self, candidates, heatmap):
        """为候选框打分：平均热度 × 面积平方根"""
        for candidate in candidates:
            avg_heat = candidate['avg_heat']
            area = candidate['area']
            # 评分公式：平均热度 × 面积平方根
            score = avg_heat * np.sqrt(area)
            candidate['score'] = score
        
        return candidates
    
    def _non_max_suppression(self, candidates, iou_threshold=0.5):
        """非极大值抑制去除重叠候选"""
        if not candidates:
            return []
        
        # 按分数降序排序
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        
        selected = []
        
        while candidates:
            # 选择得分最高的候选
            best = candidates.pop(0)
            selected.append(best)
            
            # 移除与当前最佳候选IoU过高的候选
            remaining = []
            for candidate in candidates:
                iou = self._calculate_iou(best['bbox'], candidate['bbox'])
                if iou < iou_threshold:
                    remaining.append(candidate)
            
            candidates = remaining
        
        return selected
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_debug_visualization(self, image, pred_bbox, gt_bbox, prompt, heatmap):
        """创建调试可视化图像，包含原图+框、热图、预测mask的组合（改进版）"""
        try:
            # 转换图像格式
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # 创建绘制对象
            draw = ImageDraw.Draw(pil_image)
            
            # 尝试加载更大的字体
            try:
                font = ImageFont.truetype("arial.ttf", 24)  # 增大字体
                prompt_font = ImageFont.truetype("arial.ttf", 20)  # prompt字体
            except:
                font = ImageFont.load_default()
                prompt_font = ImageFont.load_default()
            
            # 绘制预测框（红色）
            if pred_bbox is not None:
                x1, y1, x2, y2 = pred_bbox
                draw.rectangle([x1, y1, x2, y2], outline=self.pred_color, width=4)  # 增加线宽
                draw.text((x1, y1-30), "Pred", fill=self.pred_color, font=font)
            
            # 绘制真实框（绿色）
            if gt_bbox is not None:
                x1, y1, x2, y2 = gt_bbox
                draw.rectangle([x1, y1, x2, y2], outline=self.gt_color, width=4)  # 增加线宽
                draw.text((x1, y1-60), "GT", fill=self.gt_color, font=font)
            
            # 改进prompt显示 - 放在图像顶部，背景半透明
            if prompt:
                img_width, img_height = pil_image.size
                
                # 创建半透明背景
                overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # 计算文本尺寸
                prompt_text = f"Prompt: {prompt}"
                bbox = overlay_draw.textbbox((0, 0), prompt_text, font=prompt_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 绘制半透明背景矩形
                overlay_draw.rectangle([5, 5, text_width + 15, text_height + 15], 
                                     fill=(0, 0, 0, 128))  # 半透明黑色背景
                
                # 合并overlay到主图像
                pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
                
                # 重新创建draw对象
                draw = ImageDraw.Draw(pil_image)
                
                # 绘制白色文本
                draw.text((10, 10), prompt_text, fill=(255, 255, 255), font=prompt_font)
            
            # 转换回numpy数组
            result_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 创建热图可视化
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # 调整尺寸使其一致
            h, w = result_with_boxes.shape[:2]
            heatmap_colored = cv2.resize(heatmap_colored, (w, h))
            
            # 创建组合图像：左边是原图+框，右边是热图
            combined = np.hstack([result_with_boxes, heatmap_colored])
            
            return combined
            
        except Exception as e:
            print(f"创建调试可视化失败: {e}")
            return image
    
    def process_single_sample(self, sample_data):
        """处理单个样本（增强调试版本）"""
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
            
            # 加载真实bbox
            gt_bbox_xywh = self.load_ground_truth_mask(image_id, ann_id)
            if gt_bbox_xywh is None:
                return None  # 静默跳过没有对应ann_id的样本
            
            # 转换bbox格式：从[x,y,w,h]转换为[x1,y1,x2,y2]
            x, y, w, h = gt_bbox_xywh
            gt_bbox = [int(x), int(y), int(x + w), int(y + h)]
            
            # 生成预测mask和热图
            pred_mask, heatmap = self.generate_predicted_mask_with_heatmap(str(img_path), prompt)
            pred_bbox = self.mask_to_bbox(pred_mask) if pred_mask is not None else None
            
            # 创建调试可视化图像
            debug_image = self.create_debug_visualization(image, pred_bbox, gt_bbox, prompt, heatmap)
            
            return {
                'image_id': image_id,
                'ann_id': ann_id,
                'prompt': prompt,
                'result_image': debug_image,  # 这是组合的调试图像
                'pred_bbox': pred_bbox,
                'gt_bbox': gt_bbox,
                'heatmap': heatmap  # 原始热图数据
            }
            
        except Exception as e:
            print(f"处理样本异常: {e} - image_id: {sample_data.get('image_id')}, ann_id: {sample_data.get('ann_id')}")
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

    def collect_all_samples(self):
        """收集所有样本数据"""
        all_samples = []
        
        # 从texts目录读取所有.txt文件
        for text_file in self.texts_dir.glob("*.txt"):
            samples = self._read_json_lines(text_file)
            for sample in samples:
                # 添加image_id（从文件名提取）
                sample['image_id'] = text_file.stem
                all_samples.append(sample)
        
        return all_samples
    
    def generate_random_visualizations(self, num_samples=1000):
        """生成随机样本的可视化"""
        print(f"开始生成 {num_samples} 个样本的可视化...")
        
        # 收集所有样本
        all_samples = self.collect_all_samples()
        print(f"总共找到 {len(all_samples)} 个样本")
        
        if len(all_samples) == 0:
            print("没有找到任何样本数据")
            return []
        
        # 随机选择样本
        selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        results = []
        successful_count = 0
        
        for i, sample in enumerate(selected_samples, 1):
            print(f"处理样本 {i}/{len(selected_samples)}: image_id={sample.get('image_id')}, ann_id={sample.get('ann_id')}")
            
            # 处理单个样本
            result = self.process_single_sample(sample)
            
            if result is not None:
                # 保存结果图像
                output_filename = f"{sample['image_id']}_{sample['ann_id']}_debug.png"
                output_path = self.output_dir / output_filename
                
                # 保存组合的调试图像
                cv2.imwrite(str(output_path), result['result_image'])
                
                results.append({
                    'image_id': result['image_id'],
                    'ann_id': result['ann_id'],
                    'prompt': result['prompt'],
                    'output_path': str(output_path),
                    'has_pred_bbox': result['pred_bbox'] is not None,
                    'has_gt_bbox': result['gt_bbox'] is not None
                })
                
                successful_count += 1
                
                if successful_count % 10 == 0:
                    print(f"已成功处理 {successful_count} 个样本")
        
        print(f"\n可视化完成！")
        print(f"成功处理: {successful_count}/{len(selected_samples)} 个样本")
        print(f"输出目录: {self.output_dir}")
        
        # 生成汇总报告
        self.generate_summary_report(results)
        
        return results

# 使用示例
if __name__ == "__main__":
    # 创建可视化器（自动从config.yaml加载所有路径配置）
    visualizer = SimpleMaskVisualizer(
        config_path="config.yaml",
        output_dir="mask_visualization_output"
    )
    
    # 生成1000个随机样本的可视化（直接从texts目录读取所有数据）
    results = visualizer.generate_random_visualizations(num_samples=1000)