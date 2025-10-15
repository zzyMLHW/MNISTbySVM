import os
import numpy as np
from PIL import Image
from typing import Tuple

class MNISTFolderLoader:
    """从文件夹结构加载 MNIST：根目录下应为 0-9 十个子目录。"""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.class_names = [str(i) for i in range(10)]

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        data: list[np.ndarray] = []
        labels: list[int] = []
        for cls in self.class_names:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith('.bmp'):
                    continue
                fpath = os.path.join(cls_dir, fname)
                img = Image.open(fpath).convert('L')  # 灰度
                arr = np.array(img, dtype=np.float32) / 255.0
                data.append(arr.reshape(-1))
                labels.append(int(cls))
        X = np.stack(data, axis=0)
        y = np.array(labels, dtype=np.int64)
        return X, y
