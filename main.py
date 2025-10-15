import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_TEST_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_FIG_DIR,
    TRAIN_VALID_RATIO,
    RANDOM_STATE,
    SVM_PARAMS,
)
from src.data_loader import MNISTFolderLoader
from src.libsvm_trainer import LIBSVMTrainer
from src.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MNIST 多分类 SVM（LIBSVM）训练与评估')
    parser.add_argument('--train_dir', type=str, default=DEFAULT_TRAIN_DIR, help='训练集根目录')
    parser.add_argument('--test_dir', type=str, default=DEFAULT_TEST_DIR, help='测试集根目录')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='模型保存路径')
    parser.add_argument('--fig_dir', type=str, default=DEFAULT_FIG_DIR, help='图像输出目录')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) 加载数据
    train_loader = MNISTFolderLoader(args.train_dir)
    X, y = train_loader.load()
    test_loader = MNISTFolderLoader(args.test_dir)
    X_test, y_test = test_loader.load()

    # 2) 训练/验证划分
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=1 - TRAIN_VALID_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    # 3) 训练（直接使用 LIBSVM）
    print("=" * 60)
    print("开始训练阶段")
    print("=" * 60)
    
    clf = LIBSVMTrainer(**SVM_PARAMS)
    clf.fit(X_train, y_train, X_valid, y_valid, verbose=True)
    
    # 输出训练信息
    training_info = clf.get_training_info()
    print(f"训练信息总结:")
    print(f"  - 训练时间: {training_info['training_time']:.2f} 秒")
    print(f"  - 支持向量数: {training_info['support_vectors_count']}")
    print(f"  - 迭代次数: {training_info['n_iter']}")
    print("=" * 60)

    # 4) 验证与测试评估
    print("开始评估阶段")
    print("=" * 60)
    
    evaluator = Evaluator(args.fig_dir)

    print("验证集评估中...")
    y_valid_pred = clf.predict(X_valid)
    eval_valid = evaluator.evaluate(y_valid, y_valid_pred)

    print("测试集评估中...")
    y_test_pred = clf.predict(X_test)
    eval_test = evaluator.evaluate(y_test, y_test_pred)

    # 5) 保存模型
    print("保存模型中...")
    clf.save(args.model_path)

    # 6) 输出最终结果
    print("=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"验证集准确率: {eval_valid.accuracy:.4f}")
    print(f"测试集准确率: {eval_test.accuracy:.4f}")
    print(f"微平均 AUC: {eval_test.micro_auc:.4f}")
    print(f"宏平均 AUC: {eval_test.macro_auc:.4f}")
    print(f"各类别 AUC: {eval_test.class_roc_auc}")
    print(f"模型已保存至: {args.model_path}")
    print(f"评估图像已保存至: {args.fig_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()


