import os
import json
from PIL import Image
from torch.utils.data import Dataset
import re

class RefCOCODataset(Dataset):
    """
    一个用于加载和解析RefCOCO数据集的Dataset类。
    根据指定的目录结构，将图像、文本描述和边界框一一对应。
    """
    def __init__(self, data_root, image_subdir='mscoco/train2014'):
        """
        初始化数据集。

        Args:
            data_root (str): 包含 'images', 'text', 'instances' 的根目录。
            image_subdir (str): 'images' 目录下的具体子目录。
        """
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'images', image_subdir)
        self.text_dir = os.path.join(data_root, 'text')
        self.instance_dir = os.path.join(data_root, 'instances')

        self.samples = []
        self._load_data()

    def _load_data(self):
        """
        核心函数：遍历文件，加载并匹配数据，生成样本列表。
        一个样本 = (image_path, text_sentence, bbox)
        """
        print("开始加载RefCOCO数据...")
        
        # 1. 创建一个从 image_id 到 image_path 的映射
        # COCO_train2014_000000123456.jpg -> 123456
        self.img_id_to_path = {}
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.jpg'):
                # 使用正则表达式提取ID，更稳健
                match = re.search(r'_(\d+)\.jpg$', filename)
                if match:
                    img_id = int(match.group(1))
                    self.img_id_to_path[img_id] = os.path.join(self.image_dir, filename)

        # 2. 遍历text或instance目录来构建样本
        # 它们的文件名（即image_id）应该是一致的
        for ann_filename in os.listdir(self.text_dir):
            if not ann_filename.endswith('.txt'):
                continue

            img_id = int(os.path.splitext(ann_filename)[0])
            
            # 检查该ID对应的图片是否存在
            if img_id not in self.img_id_to_path:
                continue

            image_path = self.img_id_to_path[img_id]
            instance_filepath = os.path.join(self.instance_dir, ann_filename)
            text_filepath = os.path.join(self.text_dir, ann_filename)

            # 3. 读取并解析instance文件，创建 ann_id -> bbox 的映射
            ann_id_to_bbox = {}
            with open(instance_filepath, 'r') as f:
                for line in f:
                    try:
                        # 你的数据似乎是每行一个JSON对象
                        inst_data = json.loads(line.strip().replace("'", '"')) # 替换单引号以防万一
                        ann_id_to_bbox[inst_data['ann_id']] = inst_data['bbox']
                    except (json.JSONDecodeError, KeyError):
                        # 忽略格式不正确的行
                        continue

            # 4. 读取text文件，与bbox匹配，创建样本
            with open(text_filepath, 'r') as f:
                for line in f:
                    try:
                        text_data = json.loads(line.strip().replace("'", '"'))
                        ann_id = text_data['ann_id']
                        sentence = text_data['sent']

                        if ann_id in ann_id_to_bbox:
                            bbox = ann_id_to_bbox[ann_id]
                            # 添加一个完整的样本
                            self.samples.append({
                                'image_path': image_path,
                                'text': sentence,
                                'bbox': bbox # 格式是 [x, y, width, height]
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        print(f"数据加载完成！共找到 {len(self.samples)} 个 (图-文-框) 样本。")

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引获取一个样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            tuple: (image, text, bbox)
        """
        sample = self.samples[idx]
        image_path = sample['image_path']
        
        # 加载图片并确保是RGB格式
        image = Image.open(image_path).convert('RGB')
        
        text = sample['text']
        bbox = sample['bbox']
        
        return image, text, bbox

# --- 如何使用这个类 ---
# if __name__ == '__main__':
#     data_root = './data' # 你的数据根目录
#     dataset = RefCOCODataset(data_root=data_root)
#
#     # 检查一下数据加载是否正确
#     if len(dataset) > 0:
#         print("\n--- 抽查一个样本 ---")
#         image, text, bbox = dataset[0]
#         print(f"图片尺寸: {image.size}")
#         print(f"文本描述: {text}")
#         print(f"边界框 (x, y, w, h): {bbox}")
#         # image.show() # 可以取消注释来显示图片