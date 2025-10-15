import os

# 数据与输出默认路径（可通过命令行覆盖）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DIR = os.path.join(PROJECT_ROOT, 'MNIST', 'train')
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, 'MNIST', 'test')
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'svm_libsvm.joblib')
DEFAULT_FIG_DIR = os.path.join(PROJECT_ROOT, 'reports')

# 训练设置
RANDOM_STATE = 0
TRAIN_VALID_RATIO = 0.95

# LIBSVM 参数
SVM_PARAMS = {
    'C': 1.0,
    'gamma': 'scale',
    'kernel': 'rbf',
    'probability': True,
}
