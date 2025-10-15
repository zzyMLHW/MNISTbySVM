#!/usr/bin/env python3
"""
诊断 LIBSVM 多分类问题
"""
import numpy as np
from src.data_loader import MNISTFolderLoader

def debug_data():
    """检查数据格式和类别分布"""
    print("检查 MNIST 数据...")
    
    # 加载少量数据进行测试
    loader = MNISTFolderLoader('MNIST/train')
    X, y = loader.load()
    
    print(f"数据形状: {X.shape}")
    print(f"标签范围: {y.min()} - {y.max()}")
    print(f"唯一标签: {sorted(np.unique(y))}")
    print(f"标签数量: {len(np.unique(y))}")
    
    # 检查每个类别的样本数
    for label in sorted(np.unique(y)):
        count = np.sum(y == label)
        print(f"类别 {label}: {count} 个样本")
    
    # 检查数据范围
    print(f"数据范围: {X.min():.4f} - {X.max():.4f}")
    print(f"数据均值: {X.mean():.4f}")
    print(f"数据标准差: {X.std():.4f}")
    
    # 检查是否有异常值
    if X.max() > 1.0 or X.min() < 0.0:
        print("警告: 数据范围异常！")
    
    return X, y

def test_small_data():
    """用小数据集测试 LIBSVM"""
    print("\n测试小数据集...")
    
    # 创建简单的测试数据
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100个样本，10个特征
    y = np.random.randint(0, 3, 100)  # 3个类别
    
    print(f"测试数据形状: {X.shape}")
    print(f"测试标签: {sorted(np.unique(y))}")
    
    # 保存为 LIBSVM 格式
    with open('test_data.libsvm', 'w') as f:
        for i in range(len(X)):
            line = f"{y[i]}"
            for j, val in enumerate(X[i]):
                if val != 0:
                    line += f" {j+1}:{val}"
            f.write(line + "\n")
    
    print("测试数据已保存为 test_data.libsvm")
    print("请手动运行: svm-train -c 1.0 -g 0.1 -t 2 -b 1 test_data.libsvm test_model")
    print("观察输出多少个分类器...")

if __name__ == '__main__':
    debug_data()
    test_small_data()
