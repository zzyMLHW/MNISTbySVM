from dataclasses import dataclass
from typing import Dict, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

@dataclass
class EvaluationResult:
    accuracy: float
    confusion_matrix: np.ndarray
    class_roc_auc: Dict[int, float]
    micro_auc: float
    macro_auc: float
    cm_path: str
    roc_micro_macro_path: str
    roc_per_class_path: str


class Evaluator:
    """评估分类模型（支持多分类 ROC 与 AUC）。"""

    def __init__(self, fig_dir: str) -> None:
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult:
        # 准确率
        acc = accuracy_score(y_true, y_pred)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        cm_path = os.path.join(self.fig_dir, 'confusion_matrix.png')
        self._plot_confusion_matrix(cm, cm_path)

        # 伪ROC曲线（基于预测结果）
        class_auc, micro_auc, macro_auc, roc_micro_macro_path, roc_per_class_path = self._plot_pseudo_roc(y_true, y_pred)

        return EvaluationResult(
            accuracy=acc,
            confusion_matrix=cm,
            class_roc_auc=class_auc,
            micro_auc=micro_auc,
            macro_auc=macro_auc,
            cm_path=cm_path,
            roc_micro_macro_path=roc_micro_macro_path,
            roc_per_class_path=roc_per_class_path,
        )

    def _plot_confusion_matrix(self, cm: np.ndarray, out_path: str) -> None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=list(range(10)), yticklabels=list(range(10)))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def _plot_roc(self, y_true_bin: np.ndarray, y_proba: np.ndarray) -> Tuple[Dict[int, float], float, float, str, str]:
        # 逐类 ROC/AUC
        fpr: Dict[int, np.ndarray] = {}
        tpr: Dict[int, np.ndarray] = {}
        roc_auc: Dict[int, float] = {}
        num_classes = y_true_bin.shape[1]
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        # macro-average
        # 取所有 fpr 的并集，再插值平均 tpr
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        auc_macro = auc(all_fpr, mean_tpr)

        # 绘制 micro/macro 曲线
        roc_micro_macro_path = os.path.join(self.fig_dir, 'roc_micro_macro.png')
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_micro, tpr_micro, label=f'micro-average ROC (AUC = {auc_micro:.4f})', linewidth=2)
        plt.plot(all_fpr, mean_tpr, label=f'macro-average ROC (AUC = {auc_macro:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Micro & Macro Average')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(roc_micro_macro_path, dpi=200)
        plt.close()

        # 绘制每类曲线
        roc_per_class_path = os.path.join(self.fig_dir, 'roc_per_class.png')
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f'class {i} (AUC = {roc_auc[i]:.4f})', linewidth=1.5)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Per Class')
        plt.legend(loc='lower right', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(roc_per_class_path, dpi=200)
        plt.close()

        return roc_auc, auc_micro, auc_macro, roc_micro_macro_path, roc_per_class_path

    def _plot_pseudo_roc(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[int, float], float, float, str, str]:
        """绘制伪ROC曲线（基于预测结果）"""
        # 将多分类转换为二分类问题（One-vs-Rest）
        y_true_bin = label_binarize(y_true, classes=list(range(10)))
        
        # 为每个类别创建伪概率
        y_proba = np.zeros((len(y_pred), 10))
        for i in range(len(y_pred)):
            y_proba[i, y_pred[i]] = 1.0  # 预测的类别概率为1，其他为0
        
        # 逐类 ROC/AUC
        fpr: Dict[int, np.ndarray] = {}
        tpr: Dict[int, np.ndarray] = {}
        roc_auc: Dict[int, float] = {}
        num_classes = 10
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        # macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        auc_macro = auc(all_fpr, mean_tpr)

        # 绘制 micro/macro 曲线
        roc_micro_macro_path = os.path.join(self.fig_dir, 'roc_micro_macro.png')
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_micro, tpr_micro, label=f'micro-average ROC (AUC = {auc_micro:.4f})', linewidth=2)
        plt.plot(all_fpr, mean_tpr, label=f'macro-average ROC (AUC = {auc_macro:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Pseudo ROC - Micro & Macro Average')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(roc_micro_macro_path, dpi=200)
        plt.close()

        # 绘制每类曲线
        roc_per_class_path = os.path.join(self.fig_dir, 'roc_per_class.png')
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f'class {i} (AUC = {roc_auc[i]:.4f})', linewidth=1.5)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Pseudo ROC - Per Class')
        plt.legend(loc='lower right', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(roc_per_class_path, dpi=200)
        plt.close()

        return roc_auc, auc_micro, auc_macro, roc_micro_macro_path, roc_per_class_path

