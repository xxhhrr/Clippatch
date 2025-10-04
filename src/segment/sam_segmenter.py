"""
SAM (Segment Anything Model) 分割模块
基于 utils/test_sam.py 实现，用于多模态视觉补丁系统
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import yaml

class SAMSegmenter:
    """SAM分割器，支持自动分割和基于bbox的分割"""
    
    def __init__(self, config_path: str = None, sam_config: dict = None):
        """
        初始化SAM分割器
        
        Args:
            config_path: 配置文件路径
            sam_config: SAM配置字典
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.sam_config = config.get('sam', {})
        elif sam_config:
            self.sam_config = sam_config
        else:
            # 默认配置
            self.sam_config = {
                'checkpoint': 'segment-anything/weights/sam_vit_h_4b8939.pth',
                'model_type': 'vit_h',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'points_per_side': 32,
                'pred_iou_thresh': 0.86,
                'stability_score_thresh': 0.92,
                'crop_n_layers': 1,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 100
            }
        
        self.device = self.sam_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_model = None
        self.mask_generator = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """加载SAM模型"""
        try:
            checkpoint = self.sam_config['checkpoint']
            model_type = self.sam_config['model_type']
            
            print(f"Loading SAM model: {model_type} from {checkpoint}")
            self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam_model.to(device=self.device)
            
            # 创建自动分割器
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=self.sam_config.get('points_per_side', 32),
                pred_iou_thresh=self.sam_config.get('pred_iou_thresh', 0.86),
                stability_score_thresh=self.sam_config.get('stability_score_thresh', 0.92),
                crop_n_layers=self.sam_config.get('crop_n_layers', 1),
                crop_n_points_downscale_factor=self.sam_config.get('crop_n_points_downscale_factor', 2),
                min_mask_region_area=self.sam_config.get('min_mask_region_area', 100),
            )
            
            # 创建预测器（用于基于bbox的分割）
            self.predictor = SamPredictor(self.sam_model)
            
            print("SAM model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise
    
    def segment_from_bbox(self, 
                         image: np.ndarray, 
                         bboxes: List[List[int]], 
                         output_dir: str = "outputs/sam",
                         image_name: str = "image",
                         save_visualization: bool = True) -> List[Dict]:
        """
        基于bbox进行SAM分割
        
        Args:
            image: 输入图像 (H, W, 3)
            bboxes: bbox列表，格式为 [[x1, y1, x2, y2], ...]
            output_dir: 输出目录
            image_name: 图像名称
            save_visualization: 是否保存可视化结果
            
        Returns:
            masks: 分割结果列表，每个元素包含mask和相关信息
        """
        if not bboxes:
            print("No bboxes provided, using automatic segmentation")
            return self.segment_automatic(image, output_dir, image_name, save_visualization)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图像
        self.predictor.set_image(image)
        
        all_masks = []
        
        for i, bbox in enumerate(bboxes):
            try:
                # 转换bbox格式为SAM需要的格式
                input_box = np.array(bbox)  # [x1, y1, x2, y2]
                
                # 进行分割
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # 处理结果
                mask = masks[0]  # 取第一个mask
                score = scores[0]
                
                # 计算mask区域
                area = np.sum(mask)
                
                # 构建结果字典
                mask_info = {
                    'segmentation': mask,
                    'area': int(area),
                    'bbox': bbox,
                    'predicted_iou': float(score),
                    'stability_score': float(score),  # 使用predicted_iou作为stability_score
                    'crop_box': bbox,
                    'mask_id': i
                }
                
                all_masks.append(mask_info)
                
            except Exception as e:
                print(f"Error processing bbox {i}: {e}")
                continue
        
        print(f"Generated {len(all_masks)} masks from {len(bboxes)} bboxes")
        
        # 保存可视化结果
        if save_visualization and all_masks:
            self._save_visualization(image, all_masks, output_dir, f"{image_name}_bbox_sam")
            self._save_individual_masks(image, all_masks, output_dir, f"{image_name}_bbox")
        
        return all_masks
    
    def segment_automatic(self, 
                         image: np.ndarray, 
                         output_dir: str = "outputs/sam",
                         image_name: str = "image",
                         save_visualization: bool = True) -> List[Dict]:
        """
        自动分割整个图像
        
        Args:
            image: 输入图像 (H, W, 3)
            output_dir: 输出目录
            image_name: 图像名称
            save_visualization: 是否保存可视化结果
            
        Returns:
            masks: 分割结果列表
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print("Performing automatic segmentation...")
        masks = self.mask_generator.generate(image)
        print(f"Generated {len(masks)} masks automatically")
        
        # 保存可视化结果
        if save_visualization:
            self._save_visualization(image, masks, output_dir, f"{image_name}_auto_sam")
            self._save_individual_masks(image, masks, output_dir, f"{image_name}_auto")
        
        return masks
    
    def _save_visualization(self, image: np.ndarray, masks: List[Dict], 
                          output_dir: str, filename: str):
        """保存分割结果可视化"""
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            self._show_masks(masks)
            plt.axis('off')
            plt.title(f"SAM Segmentation Results ({len(masks)} masks)")
            
            save_path = os.path.join(output_dir, f"{filename}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            print(f"Visualization saved to: {save_path}")
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def _save_individual_masks(self, image: np.ndarray, masks: List[Dict], 
                             output_dir: str, prefix: str):
        """保存单个mask"""
        try:
            masks_dir = os.path.join(output_dir, f"{prefix}_masks")
            os.makedirs(masks_dir, exist_ok=True)
            
            for i, mask_info in enumerate(masks):
                mask = mask_info['segmentation']
                
                # 保存mask为二值图像
                mask_img = (mask * 255).astype(np.uint8)
                mask_path = os.path.join(masks_dir, f"mask_{i:03d}.png")
                cv2.imwrite(mask_path, mask_img)
                
                # 保存带颜色的mask叠加图像
                colored_mask = image.copy()
                color = np.random.randint(0, 255, 3)
                colored_mask[mask] = colored_mask[mask] * 0.6 + color * 0.4
                
                colored_path = os.path.join(masks_dir, f"colored_mask_{i:03d}.png")
                cv2.imwrite(colored_path, cv2.cvtColor(colored_mask.astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            print(f"Individual masks saved to: {masks_dir}")
            
        except Exception as e:
            print(f"Error saving individual masks: {e}")
    
    def _show_masks(self, masks: List[Dict], ax=None):
        """显示masks（基于test_sam.py的show_anns函数）"""
        if len(masks) == 0:
            return
        
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        if ax is None:
            ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_masks[0]['segmentation'].shape[0],
                       sorted_masks[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        
        for mask_info in sorted_masks:
            m = mask_info['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        
        ax.imshow(img)
    
    def get_mask_statistics(self, masks: List[Dict]) -> Dict:
        """获取mask统计信息"""
        if not masks:
            return {}
        
        areas = [mask['area'] for mask in masks]
        scores = [mask.get('predicted_iou', mask.get('stability_score', 0)) for mask in masks]
        
        stats = {
            'total_masks': len(masks),
            'total_area': sum(areas),
            'avg_area': np.mean(areas),
            'median_area': np.median(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'avg_score': np.mean(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0
        }
        
        return stats


def create_sam_segmenter(config_path: str = None) -> SAMSegmenter:
    """创建SAM分割器的工厂函数"""
    return SAMSegmenter(config_path=config_path)


if __name__ == "__main__":
    # 测试代码
    segmenter = SAMSegmenter()
    
    # 测试图像路径
    test_image_path = "data/worldcup.jpg"  # 替换为实际路径
    
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 测试自动分割
        masks = segmenter.segment_automatic(image, "outputs/sam", "test")
        
        # 打印统计信息
        stats = segmenter.get_mask_statistics(masks)
        print("Mask Statistics:", stats)
    else:
        print(f"Test image not found: {test_image_path}")