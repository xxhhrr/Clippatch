import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils.clip_util import get_heatmap
import os

def test_heatmap_fix():
    """测试热图修复效果"""
    # 固定的测试用例
    test_cases = [
        {
            "image_path": "../data/images/mscoco/train2014/COCO_train2014_000000095023.jpg",
            "text_prompt": "pears"
        },
        {
            "image_path": "../data/images/mscoco/train2014/COCO_train2014_000000000144.jpg", 
            "text_prompt": "middle giraffe"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        image_path = test_case["image_path"]
        text_prompt = test_case["text_prompt"]
        
        print(f"测试 {i}: {os.path.basename(image_path)} - '{text_prompt}'")
        
        if not os.path.exists(image_path):
            print(f"  跳过：文件不存在 - {image_path}")
            continue
        
        try:
            # 使用现有的get_heatmap方法生成热图
            heatmap = get_heatmap(image_path, text_prompt)
            
            # 加载原图
            image = Image.open(image_path).convert('RGB')
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原图
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 热图
            heatmap_colored = plt.cm.jet(heatmap)
            axes[1].imshow(heatmap_colored)
            axes[1].set_title(f'Heatmap: "{text_prompt}"')
            axes[1].axis('off')
            
            # 叠加图
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            
            # 创建叠加效果
            heatmap_colored_resized = plt.cm.jet(heatmap_resized)[:,:,:3]
            overlay = 0.6 * img_array/255.0 + 0.4 * heatmap_colored_resized
            overlay = np.clip(overlay, 0, 1)
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # 统计信息
            high_activation_ratio = np.sum(heatmap > 0.7) / heatmap.size
            print(f"  高激活区域比例: {high_activation_ratio:.3f}")
            
        except Exception as e:
            print(f"  测试失败: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("简化热图测试")
    print("=" * 30)
    test_heatmap_fix()