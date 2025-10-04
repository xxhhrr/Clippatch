"""
多模态视觉补丁系统主管道
集成 CLIP定位 -> SAM分割 -> 压缩传输 -> 重建 -> 评估
"""

import os
import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入各模块
from src.segment.sam_segmenter import SAMSegmenter
from src.compress.semantic_packer import SemanticPacker

class MultiModalPatchPipeline:
    """多模态视觉补丁系统主管道"""
    
    def __init__(self, config_path: str):
        """
        初始化管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化各模块
        self.sam_segmenter = None
        self.compressor = None
        
        # 创建输出目录
        self._setup_directories()
        
        # 初始化模块
        self._initialize_modules()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """创建必要的输出目录"""
        paths = self.config.get('paths', {})
        
        directories = [
            paths.get('output_root', 'outputs'),
            paths.get('logs', 'outputs/logs'),
            paths.get('sam_output', 'outputs/sam'),
            paths.get('compress_output', 'outputs/payloads'),
            paths.get('reconstruction_output', 'outputs/recon')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_modules(self):
        """初始化各个模块"""
        print("Initializing modules...")
        
        # 初始化SAM分割器
        try:
            self.sam_segmenter = SAMSegmenter(config_path=self.config_path)
            print("✓ SAM segmenter initialized")
        except Exception as e:
            print(f"✗ Failed to initialize SAM segmenter: {e}")
            self.sam_segmenter = None
        
        # 初始化压缩器
        try:
            compress_config = self.config.get('compress', {})
            self.compressor = SemanticPacker(
                method=compress_config.get('method', 'semantic'),
                quality=compress_config.get('quality', 85),
                max_size=tuple(compress_config.get('max_size', [512, 512]))
            )
            print("✓ Compressor initialized")
        except Exception as e:
            print(f"✗ Failed to initialize compressor: {e}")
            self.compressor = None
    
    def locate_targets(self, image: np.ndarray, text_query: str) -> List[List[int]]:
        """
        使用CLIP定位目标区域（占位符实现）
        
        Args:
            image: 输入图像
            text_query: 文本查询
            
        Returns:
            bboxes: 目标区域bbox列表 [[x1, y1, x2, y2], ...]
        """
        # TODO: 集成rl目录中的CLIP定位功能
        # 这里先返回一个示例bbox
        h, w = image.shape[:2]
        
        # 示例：返回图像中心区域的bbox
        center_x, center_y = w // 2, h // 2
        bbox_size = min(w, h) // 3
        
        bbox = [
            center_x - bbox_size // 2,
            center_y - bbox_size // 2,
            center_x + bbox_size // 2,
            center_y + bbox_size // 2
        ]
        
        print(f"Located target region: {bbox} for query: '{text_query}'")
        return [bbox]
    
    def segment_targets(self, image: np.ndarray, bboxes: List[List[int]], 
                       image_name: str = "image") -> List[Dict]:
        """
        使用SAM分割目标区域
        
        Args:
            image: 输入图像
            bboxes: 目标区域bbox列表
            image_name: 图像名称
            
        Returns:
            masks: 分割结果列表
        """
        if not self.sam_segmenter:
            print("SAM segmenter not available")
            return []
        
        sam_config = self.config.get('sam', {})
        output_dir = self.config['paths']['sam_output']
        
        masks = self.sam_segmenter.segment_from_bbox(
            image=image,
            bboxes=bboxes,
            output_dir=output_dir,
            image_name=image_name,
            save_visualization=sam_config.get('save_visualization', True)
        )
        
        # 打印统计信息
        stats = self.sam_segmenter.get_mask_statistics(masks)
        print(f"SAM segmentation completed: {stats}")
        
        return masks
    
    def compress_masks(self, masks: List[Dict], image_name: str = "image") -> List[bytes]:
        """
        压缩mask数据
        
        Args:
            masks: 分割结果列表
            image_name: 图像名称
            
        Returns:
            compressed_data: 压缩后的数据列表
        """
        if not self.compressor or not masks:
            return []
        
        output_dir = self.config['paths']['compress_output']
        compressed_data = []
        
        for i, mask_info in enumerate(masks):
            mask = mask_info['segmentation']
            
            # 压缩mask
            compressed = self.compressor.compress_mask(
                mask, 
                f"{image_name}_mask_{i:03d}",
                output_dir
            )
            
            if compressed:
                compressed_data.append(compressed)
        
        print(f"Compressed {len(compressed_data)} masks")
        return compressed_data
    
    def reconstruct_masks(self, compressed_data: List[bytes], 
                         image_name: str = "image") -> List[np.ndarray]:
        """
        重建mask数据
        
        Args:
            compressed_data: 压缩数据列表
            image_name: 图像名称
            
        Returns:
            reconstructed_masks: 重建的mask列表
        """
        if not self.compressor or not compressed_data:
            return []
        
        output_dir = self.config['paths']['reconstruction_output']
        reconstructed_masks = []
        
        for i, data in enumerate(compressed_data):
            # 解压缩mask
            reconstructed = self.compressor.decompress_mask(
                data,
                f"{image_name}_recon_{i:03d}",
                output_dir
            )
            
            if reconstructed is not None:
                reconstructed_masks.append(reconstructed)
        
        print(f"Reconstructed {len(reconstructed_masks)} masks")
        return reconstructed_masks
    
    def evaluate_results(self, original_masks: List[Dict], 
                        reconstructed_masks: List[np.ndarray]) -> Dict:
        """
        评估重建结果
        
        Args:
            original_masks: 原始mask列表
            reconstructed_masks: 重建mask列表
            
        Returns:
            evaluation_results: 评估结果
        """
        # TODO: 实现详细的评估指标
        results = {
            'original_count': len(original_masks),
            'reconstructed_count': len(reconstructed_masks),
            'success_rate': len(reconstructed_masks) / max(len(original_masks), 1)
        }
        
        print(f"Evaluation results: {results}")
        return results
    
    def process_image(self, image_path: str, text_query: str, 
                     output_prefix: str = None) -> Dict:
        """
        处理单张图像的完整流程
        
        Args:
            image_path: 图像路径
            text_query: 文本查询
            output_prefix: 输出前缀
            
        Returns:
            results: 处理结果
        """
        # 读取图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_name = output_prefix or Path(image_path).stem
        
        print(f"\n{'='*50}")
        print(f"Processing: {image_path}")
        print(f"Query: {text_query}")
        print(f"Image shape: {image.shape}")
        print(f"{'='*50}")
        
        results = {}
        
        try:
            # Step 1: CLIP定位
            print("\n1. Locating targets with CLIP...")
            bboxes = self.locate_targets(image, text_query)
            results['bboxes'] = bboxes
            
            # Step 2: SAM分割
            print("\n2. Segmenting with SAM...")
            masks = self.segment_targets(image, bboxes, image_name)
            results['masks'] = masks
            
            # Step 3: 压缩传输
            print("\n3. Compressing masks...")
            compressed_data = self.compress_masks(masks, image_name)
            results['compressed_data'] = compressed_data
            
            # Step 4: 重建
            print("\n4. Reconstructing masks...")
            reconstructed_masks = self.reconstruct_masks(compressed_data, image_name)
            results['reconstructed_masks'] = reconstructed_masks
            
            # Step 5: 评估
            print("\n5. Evaluating results...")
            evaluation = self.evaluate_results(masks, reconstructed_masks)
            results['evaluation'] = evaluation
            
            print(f"\n✓ Processing completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_batch(self, image_dir: str, queries: List[str]) -> List[Dict]:
        """
        批量处理图像
        
        Args:
            image_dir: 图像目录
            queries: 查询列表
            
        Returns:
            all_results: 所有结果列表
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        all_results = []
        
        for image_file in image_files:
            for query in queries:
                output_prefix = f"{image_file.stem}_{len(all_results):03d}"
                
                try:
                    result = self.process_image(str(image_file), query, output_prefix)
                    result['image_path'] = str(image_file)
                    result['query'] = query
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error processing {image_file} with query '{query}': {e}")
                    all_results.append({
                        'image_path': str(image_file),
                        'query': query,
                        'error': str(e)
                    })
        
        return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态视觉补丁系统")
    parser.add_argument('--config', type=str, default='configs/mvp.yaml',
                       help='配置文件路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径')
    parser.add_argument('--query', type=str, default='person',
                       help='文本查询')
    parser.add_argument('--queries', type=str, nargs='+',
                       help='多个文本查询')
    
    args = parser.parse_args()
    
    # 创建管道
    pipeline = MultiModalPatchPipeline(args.config)
    
    if args.image:
        # 处理单张图像
        queries = args.queries if args.queries else [args.query]
        
        for query in queries:
            result = pipeline.process_image(args.image, query)
            print(f"\nResult: {result}")
    
    elif args.image_dir:
        # 批量处理
        queries = args.queries if args.queries else [args.query]
        results = pipeline.run_batch(args.image_dir, queries)
        
        print(f"\nProcessed {len(results)} image-query pairs")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"outputs/logs/batch_results_{timestamp}.yaml"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Results saved to: {results_file}")
    
    else:
        print("Please provide either --image or --image_dir")


if __name__ == "__main__":
    main()