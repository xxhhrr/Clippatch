import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils.clip_util import get_heatmap
import os

def clean_heatmap_noise(heatmap, method='threshold', **kwargs):
    """清理热图噪声的多种方法"""
    if method == 'threshold':
        threshold = kwargs.get('threshold', 0.15)
        cleaned = np.where(heatmap < threshold, 0, heatmap)
        if cleaned.max() > 0:
            cleaned = (cleaned - cleaned.min()) / (cleaned.max() + 1e-6)
        return cleaned
    
    elif method == 'percentile':
        percentile = kwargs.get('percentile', 70)
        flat_heat = heatmap.flatten()
        nonzero_heat = flat_heat[flat_heat > 0]
        if len(nonzero_heat) > 0:
            threshold = np.percentile(nonzero_heat, percentile)
            cleaned = np.where(heatmap < threshold, 0, heatmap)
            if cleaned.max() > 0:
                cleaned = (cleaned - cleaned.min()) / (cleaned.max() + 1e-6)
            return cleaned
        return heatmap
    
    elif method == 'gaussian_threshold':
        # 高斯滤波 + 阈值
        sigma = kwargs.get('sigma', 0.5)
        threshold = kwargs.get('threshold', 0.2)
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter(heatmap, sigma=sigma)
        cleaned = np.where(smoothed < threshold, 0, smoothed)
        if cleaned.max() > 0:
            cleaned = (cleaned - cleaned.min()) / (cleaned.max() + 1e-6)
        return cleaned
    
    elif method == 'morphology':
        # 形态学操作去噪
        threshold = kwargs.get('threshold', 0.1)
        kernel_size = kwargs.get('kernel_size', 3)
        
        # 先阈值化
        binary = (heatmap > threshold).astype(np.uint8)
        
        # 形态学开运算去除小噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 应用到原热图
        cleaned = heatmap * cleaned_binary
        if cleaned.max() > 0:
            cleaned = (cleaned - cleaned.min()) / (cleaned.max() + 1e-6)
        return cleaned
    
    return heatmap

def debug_heatmap_processing():
    """逐步调试热图处理过程，重点测试噪声消除"""
    # 测试用例
    image_path = "../data/images/mscoco/train2014/COCO_train2014_000000095023.jpg"
    text_prompt = "pears"
    
    print(f"调试热图处理: {os.path.basename(image_path)} - '{text_prompt}'")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return
    
    try:
        # 步骤1: 获取原始热图
        print("步骤1: 获取原始热图数据")
        heatmap_raw = get_heatmap(image_path, text_prompt)
        print(f"  原始热图形状: {heatmap_raw.shape}")
        print(f"  数值范围: [{heatmap_raw.min():.4f}, {heatmap_raw.max():.4f}]")
        print(f"  数据类型: {heatmap_raw.dtype}")
        
        # 分析原始热图的数值分布
        print(f"  零值像素数: {(heatmap_raw == 0).sum()}")
        print(f"  微小值像素数(<0.1): {(heatmap_raw < 0.1).sum()}")
        print(f"  中等值像素数(0.1-0.5): {((heatmap_raw >= 0.1) & (heatmap_raw < 0.5)).sum()}")
        print(f"  高值像素数(>=0.5): {(heatmap_raw >= 0.5).sum()}")
        
        # 加载原图
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        heatmap_resized = cv2.resize(heatmap_raw, (w, h))
        
        # 创建显示窗口 - 扩展为4x4布局
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        fig.suptitle(f'热图噪声消除测试: {text_prompt}', fontsize=16)
        
        # 第一行：原始数据
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(heatmap_resized, cmap='gray')
        axes[0, 1].set_title(f'原始热图(灰度)\n范围:[{heatmap_resized.min():.3f}, {heatmap_resized.max():.3f}]')
        axes[0, 1].axis('off')
        
        # 原始热图的cv2颜色映射（有噪声）
        heatmap_255_original = (heatmap_resized * 255).astype(np.uint8)
        color_cv2_original = cv2.applyColorMap(heatmap_255_original, cv2.COLORMAP_JET)
        color_cv2_rgb_original = cv2.cvtColor(color_cv2_original, cv2.COLOR_BGR2RGB)
        
        axes[0, 2].imshow(color_cv2_rgb_original)
        axes[0, 2].set_title('原始cv2颜色映射\n(有红色噪声)')
        axes[0, 2].axis('off')
        
        # 原始混合效果
        overlay_original = np.clip(img_array * 0.5 + color_cv2_rgb_original * 0.5, 0, 255).astype(np.uint8)
        axes[0, 3].imshow(overlay_original)
        axes[0, 3].set_title('原始混合效果\n(有噪声)')
        axes[0, 3].axis('off')
        
        # 第二行：不同的噪声消除方法
        methods = [
            ('threshold', {'threshold': 0.15}),
            ('percentile', {'percentile': 75}),
            ('gaussian_threshold', {'sigma': 0.8, 'threshold': 0.2}),
            ('morphology', {'threshold': 0.1, 'kernel_size': 3})
        ]
        
        cleaned_heatmaps = []
        
        for i, (method, params) in enumerate(methods):
            print(f"\n步骤{i+2}: 测试{method}方法去噪")
            
            # 清理噪声
            cleaned = clean_heatmap_noise(heatmap_resized, method=method, **params)
            cleaned_heatmaps.append(cleaned)
            
            print(f"  清理后范围: [{cleaned.min():.4f}, {cleaned.max():.4f}]")
            print(f"  零值像素数: {(cleaned == 0).sum()}")
            print(f"  非零像素数: {(cleaned > 0).sum()}")
            
            # 显示清理后的灰度图
            axes[1, i].imshow(cleaned, cmap='gray')
            axes[1, i].set_title(f'{method}清理后\n零值:{(cleaned == 0).sum()}')
            axes[1, i].axis('off')
        
        # 第三行：清理后的cv2颜色映射
        for i, (method, cleaned) in enumerate(zip([m[0] for m in methods], cleaned_heatmaps)):
            # 应用cv2颜色映射
            cleaned_255 = (cleaned * 255).astype(np.uint8)
            color_cv2_cleaned = cv2.applyColorMap(cleaned_255, cv2.COLORMAP_JET)
            color_cv2_rgb_cleaned = cv2.cvtColor(color_cv2_cleaned, cv2.COLOR_BGR2RGB)
            
            axes[2, i].imshow(color_cv2_rgb_cleaned)
            axes[2, i].set_title(f'{method}\ncv2颜色映射')
            axes[2, i].axis('off')
        
        # 第四行：清理后的混合效果
        for i, (method, cleaned) in enumerate(zip([m[0] for m in methods], cleaned_heatmaps)):
            # 混合显示
            cleaned_255 = (cleaned * 255).astype(np.uint8)
            color_cv2_cleaned = cv2.applyColorMap(cleaned_255, cv2.COLORMAP_JET)
            color_cv2_rgb_cleaned = cv2.cvtColor(color_cv2_cleaned, cv2.COLOR_BGR2RGB)
            
            overlay_cleaned = np.clip(img_array * 0.5 + color_cv2_rgb_cleaned * 0.5, 0, 255).astype(np.uint8)
            
            axes[3, i].imshow(overlay_cleaned)
            axes[3, i].set_title(f'{method}\n最终混合效果')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细统计
        print("\n" + "=" * 60)
        print("噪声消除效果对比:")
        print(f"原始热图: 零值={((heatmap_resized == 0).sum()):>6}, 非零值={((heatmap_resized > 0).sum()):>6}")
        
        for method, cleaned in zip([m[0] for m in methods], cleaned_heatmaps):
            zero_count = (cleaned == 0).sum()
            nonzero_count = (cleaned > 0).sum()
            high_count = (cleaned > 0.5).sum()
            print(f"{method:>15}: 零值={zero_count:>6}, 非零值={nonzero_count:>6}, 高值(>0.5)={high_count:>6}")
        
        print("\n推荐使用 'threshold' 或 'percentile' 方法，效果最稳定")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_heatmap_processing()