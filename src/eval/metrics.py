"""
Evaluation Metrics
计算CLIP(I_hat,T)、IoU/PSNR/SSIM/LPIPS和bpp指标
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from rl.utils.clip_util import get_heatmap

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

class MetricsCalculator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 初始化LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips_fn = None
    
    def calculate_all_metrics(self, original_image, reconstructed_image, 
                            text_prompt, gt_mask=None, pred_mask=None, bpp=None):
        """
        计算所有评估指标
        """
        metrics = {}
        
        # 1. CLIP相似度
        metrics['clip_similarity'] = self._calculate_clip_similarity(
            reconstructed_image, text_prompt)
        
        # 2. 图像质量指标
        if original_image is not None:
            metrics.update(self._calculate_image_quality_metrics(
                original_image, reconstructed_image))
        
        # 3. 分割质量指标
        if gt_mask is not None and pred_mask is not None:
            metrics.update(self._calculate_segmentation_metrics(
                gt_mask, pred_mask))
        
        # 4. 压缩效率
        if bpp is not None:
            metrics['bpp'] = bpp
        
        return metrics
    
    def _calculate_clip_similarity(self, image, text_prompt):
        """计算CLIP(I_hat, T)相似度"""
        try:
            # 保存临时图像
            temp_path = "temp_recon.jpg"
            if isinstance(image, np.ndarray):
                cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                image.save(temp_path)
            
            # 获取CLIP特征并计算相似度
            heatmap = get_heatmap(temp_path, text_prompt)
            clip_score = np.mean(heatmap)  # 简化的相似度计算
            
            # 清理临时文件
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return float(clip_score)
        except Exception as e:
            print(f"Error calculating CLIP similarity: {e}")
            return 0.0
    
    def _calculate_image_quality_metrics(self, original, reconstructed):
        """计算图像质量指标"""
        metrics = {}
        
        # 确保图像格式一致
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, 
                                     (original.shape[1], original.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
        
        # 转换为相同数据类型
        original = original.astype(np.float64)
        reconstructed = reconstructed.astype(np.float64)
        
        try:
            # PSNR
            metrics['psnr'] = psnr(original, reconstructed, data_range=255.0)
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            metrics['psnr'] = 0.0
        
        try:
            # SSIM
            if len(original.shape) == 3:
                metrics['ssim'] = ssim(original, reconstructed, 
                                     multichannel=True, channel_axis=2,
                                     data_range=255.0)
            else:
                metrics['ssim'] = ssim(original, reconstructed, 
                                     data_range=255.0)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            metrics['ssim'] = 0.0
        
        # LPIPS
        if self.lpips_fn is not None:
            try:
                # 转换为tensor并归一化到[-1, 1]
                orig_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float()
                recon_tensor = torch.from_numpy(reconstructed).permute(2, 0, 1).unsqueeze(0).float()
                
                orig_tensor = (orig_tensor / 127.5) - 1.0
                recon_tensor = (recon_tensor / 127.5) - 1.0
                
                orig_tensor = orig_tensor.to(self.device)
                recon_tensor = recon_tensor.to(self.device)
                
                with torch.no_grad():
                    lpips_score = self.lpips_fn(orig_tensor, recon_tensor)
                    metrics['lpips'] = float(lpips_score.cpu().item())
            except Exception as e:
                print(f"Error calculating LPIPS: {e}")
                metrics['lpips'] = 1.0
        else:
            metrics['lpips'] = None
        
        return metrics
    
    def _calculate_segmentation_metrics(self, gt_mask, pred_mask):
        """计算分割质量指标"""
        metrics = {}
        
        # 确保mask格式一致
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                 (gt_mask.shape[1], gt_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        
        # 二值化
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)
        
        # IoU
        intersection = np.sum(gt_binary & pred_binary)
        union = np.sum(gt_binary | pred_binary)
        
        if union > 0:
            metrics['iou'] = intersection / union
        else:
            metrics['iou'] = 1.0 if intersection == 0 else 0.0
        
        # Dice coefficient
        if np.sum(gt_binary) + np.sum(pred_binary) > 0:
            metrics['dice'] = 2 * intersection / (np.sum(gt_binary) + np.sum(pred_binary))
        else:
            metrics['dice'] = 1.0
        
        # Precision and Recall
        if np.sum(pred_binary) > 0:
            metrics['precision'] = intersection / np.sum(pred_binary)
        else:
            metrics['precision'] = 1.0 if np.sum(gt_binary) == 0 else 0.0
        
        if np.sum(gt_binary) > 0:
            metrics['recall'] = intersection / np.sum(gt_binary)
        else:
            metrics['recall'] = 1.0
        
        return metrics
    
    def save_metrics_to_csv(self, metrics_list, output_path):
        """保存指标到CSV文件"""
        import pandas as pd
        
        df = pd.DataFrame(metrics_list)
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")

def create_metrics_calculator(config):
    """工厂函数"""
    return MetricsCalculator(config)