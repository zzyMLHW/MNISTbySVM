# MNISTbySVM

本项目使用基于 LIBSVM 的多分类 SVM（scikit-learn 的 SVC 底层为 LIBSVM）对 MNIST 手写数字进行分类训练与评估，生成准确率、混淆矩阵与 ROC 曲线，并输出 Markdown 报告。

## 依赖安装
```bash
pip install -r requirements.txt
```

## LIBSVM 安装
需要安装 LIBSVM 命令行工具：

### Ubuntu/Debian:
```bash
sudo apt-get install libsvm-tools
```

### 或者从源码编译:
```bash
wget https://github.com/cjlin1/libsvm/archive/v3.25.tar.gz
tar -xzf v3.25.tar.gz
cd libsvm-3.25
make
sudo cp svm-train svm-predict svm-scale /usr/local/bin/
```

### 验证安装:
```bash
svm-train -h
svm-predict -h
```

## 数据目录
默认从 `MNIST/train` 与 `MNIST/test` 读取数据，每个目录下有 `0-9` 十个子目录，内部为 28x28 的 `.bmp` 图像。

## 运行
```bash
python main.py --train_dir MNIST/train --test_dir MNIST/test --model_path models/svm_libsvm.joblib --fig_dir reports
```

参数：
- `--train_dir`：训练集根目录
- `--test_dir`：测试集根目录
- `--model_path`：模型保存路径
- `--fig_dir`：评估图像输出目录

## 说明
- 训练集会按 95%/5% 划分训练与验证。
- 将输出混淆矩阵与 ROC 曲线到指定目录。
- 程序会输出准确率、AUC 等评估指标。
