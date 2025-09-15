import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils.clip_util import get_heatmap
import os

def debug_heatmap_processing():
    """逐步调试热图处理过程"""
    # 测试用例
    image_path = "../data/images/mscoco/train2014/COCO_train2014_000095023.jpg"
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
        
        # 加载原图
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # 创建显示窗口
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'热图处理调试: {text_prompt}', fontsize=16)
        
        # 步骤1显示: 原图和原始热图
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(heatmap_raw, cmap='gray')
        axes[0, 1].set_title(f'原始热图(灰度)\n范围:[{heatmap_raw.min():.3f}, {heatmap_raw.max():.3f}]')
        axes[0, 1].axis('off')
        
        # 步骤2: matplotlib的jet颜色映射
        print("\n步骤2: 使用matplotlib的jet颜色映射")
        heatmap_plt_jet = plt.cm.jet(heatmap_raw)
        print(f"  plt.cm.jet后形状: {heatmap_plt_jet.shape}")
        print(f"  数值范围: [{heatmap_plt_jet.min():.4f}, {heatmap_plt_jet.max():.4f}]")
        
        axes[0, 2].imshow(heatmap_plt_jet)
        axes[0, 2].set_title('matplotlib jet颜色映射')
        axes[0, 2].axis('off')
        
        # 步骤3: 调整热图尺寸到原图大小
        print("\n步骤3: 调整热图尺寸")
        h, w = img_array.shape[:2]
        heatmap_resized = cv2.resize(heatmap_raw, (w, h))
        print(f"  调整后形状: {heatmap_resized.shape}")
        print(f"  原图尺寸: {img_array.shape}")
        
        axes[0, 3].imshow(heatmap_resized, cmap='gray')
        axes[0, 3].set_title(f'调整尺寸后(灰度)\n{heatmap_resized.shape}')
        axes[0, 3].axis('off')
        
        # 步骤4: 对调整尺寸后的热图应用jet颜色映射
        print("\n步骤4: 对调整尺寸后的热图应用颜色映射")
        heatmap_resized_colored = plt.cm.jet(heatmap_resized)
        print(f"  颜色映射后形状: {heatmap_resized_colored.shape}")
        
        axes[1, 0].imshow(heatmap_resized_colored)
        axes[1, 0].set_title('调整尺寸+jet颜色映射')
        axes[1, 0].axis('off')
        
        # 步骤5: 使用cv2的颜色映射方法(类似main.py)
        print("\n步骤5: 使用cv2颜色映射(main.py方式)")
        # 先转换为0-255范围
        heatmap_255 = (heatmap_resized * 255).astype(np.uint8)
        print(f"  转换为0-255后范围: [{heatmap_255.min()}, {heatmap_255.max()}]")
        
        color_cv2 = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
        color_cv2_rgb = cv2.cvtColor(color_cv2, cv2.COLOR_BGR2RGB)
        print(f"  cv2颜色映射后形状: {color_cv2_rgb.shape}")
        print(f"  cv2颜色映射后范围: [{color_cv2_rgb.min()}, {color_cv2_rgb.max()}]")
        
        axes[1, 1].imshow(color_cv2_rgb)
        axes[1, 1].set_title('cv2 COLORMAP_JET')
        axes[1, 1].axis('off')
        
        # 步骤6: main.py风格的混合显示
        print("\n步骤6: main.py风格的混合显示")
        overlay_main_style = np.clip(img_array * 0.5 + color_cv2_rgb * 0.5, 0, 255).astype(np.uint8)
        print(f"  main.py风格混合后形状: {overlay_main_style.shape}")
        
        axes[1, 2].imshow(overlay_main_style)
        axes[1, 2].set_title('main.py风格混合\n(原图50% + 热图50%)')
        axes[1, 2].axis('off')
        
        # 步骤7: 测试文件风格的混合显示
        print("\n步骤7: 测试文件风格的混合显示")
        heatmap_colored_resized = plt.cm.jet(heatmap_resized)[:,:,:3]  # 去掉alpha通道
        overlay_test_style = 0.6 * img_array/255.0 + 0.4 * heatmap_colored_resized
        overlay_test_style = np.clip(overlay_test_style, 0, 1)
        print(f"  测试文件风格混合后形状: {overlay_test_style.shape}")
        print(f"  测试文件风格混合后范围: [{overlay_test_style.min():.4f}, {overlay_test_style.max():.4f}]")
        
        axes[1, 3].imshow(overlay_test_style)
        axes[1, 3].set_title('测试文件风格混合\n(原图60% + 热图40%)')
        axes[1, 3].axis('off')
        
        # 步骤8: 比较不同处理方式的差异
        print("\n步骤8: 差异分析")
        
        # matplotlib vs cv2颜色映射的差异
        diff_colormap = np.abs(heatmap_resized_colored[:,:,:3] - color_cv2_rgb/255.0)
        axes[2, 0].imshow(diff_colormap)
        axes[2, 0].set_title('颜色映射差异\n(matplotlib vs cv2)')
        axes[2, 0].axis('off')
        
        # 混合方式的差异
        diff_overlay = np.abs(overlay_test_style - overlay_main_style/255.0)
        axes[2, 1].imshow(diff_overlay)
        axes[2, 1].set_title('混合方式差异\n(测试 vs main.py)')
        axes[2, 1].axis('off')
        
        # 热图统计信息
        axes[2, 2].hist(heatmap_raw.flatten(), bins=50, alpha=0.7, label='原始热图')
        axes[2, 2].hist(heatmap_resized.flatten(), bins=50, alpha=0.7, label='调整尺寸后')
        axes[2, 2].set_title('热图数值分布')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        # 显示关键统计信息
        stats_text = f"""原始热图统计:
均值: {heatmap_raw.mean():.4f}
标准差: {heatmap_raw.std():.4f}
最小值: {heatmap_raw.min():.4f}
最大值: {heatmap_raw.max():.4f}

调整尺寸后:
均值: {heatmap_resized.mean():.4f}
标准差: {heatmap_resized.std():.4f}
高激活区域(>0.7): {(heatmap_resized > 0.7).sum()}/{heatmap_resized.size}
比例: {(heatmap_resized > 0.7).sum()/heatmap_resized.size:.3f}"""
        
        axes[2, 3].text(0.05, 0.95, stats_text, transform=axes[2, 3].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[2, 3].set_title('统计信息')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印总结
        print("\n" + "=" * 60)
        print("处理步骤总结:")
        print("1. get_heatmap() 返回 [0,1] 范围的热图")
        print("2. matplotlib.cm.jet() 直接应用颜色映射")
        print("3. cv2.applyColorMap() 需要先转换为 [0,255] uint8")
        print("4. main.py 使用 cv2 + BGR到RGB转换 + 50%混合")
        print("5. 测试文件使用 matplotlib + 60%/40%混合")
        print("6. 主要差异在颜色映射方法和混合比例")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_heatmap_processing()