"""
直接使用 LIBSVM 库进行训练，支持进度监控
"""
import os
import time
import joblib
import numpy as np
from typing import Optional, Dict, Any
import subprocess
import tempfile

class LIBSVMTrainer:
    """直接使用 LIBSVM 进行训练，支持进度监控"""
    
    def __init__(self, **svm_params):
        self.params = svm_params
        self.model_path = None
        self.training_history = {
            'training_time': 0,
            'support_vectors_count': 0,
            'n_iter': 0
        }
    
    def _create_libsvm_data(self, X, y, filepath):
        """将数据转换为 LIBSVM 格式"""
        with open(filepath, 'w') as f:
            for i in range(len(X)):
                # LIBSVM 格式: label feature1:value1 feature2:value2 ...
                line = f"{y[i]}"
                for j, val in enumerate(X[i]):
                    if val != 0:  # 只保存非零特征
                        line += f" {j+1}:{val}"
                f.write(line + "\n")
    
    def _parse_libsvm_output(self, output):
        """解析 LIBSVM 输出，提取训练信息"""
        lines = output.split('\n')
        for line in lines:
            if 'optimization finished' in line:
                # 提取迭代次数
                if 'iterations' in line:
                    try:
                        self.training_history['n_iter'] = int(line.split('iterations')[0].split()[-1])
                    except:
                        pass
            elif 'Total nSV' in line:
                # 提取支持向量数
                try:
                    self.training_history['support_vectors_count'] = int(line.split('Total nSV =')[1].strip())
                except:
                    pass
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """训练 LIBSVM 模型"""
        if verbose:
            print("开始 LIBSVM 训练...")
            print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")
        
        start_time = time.time()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.libsvm', delete=False) as train_file:
            train_path = train_file.name
        
        try:
            # 转换数据格式
            if verbose:
                print("转换数据格式...")
            self._create_libsvm_data(X, y, train_path)
            
            # 构建 LIBSVM 命令
            cmd = ['svm-train']
            
            # 添加参数
            if 'C' in self.params:
                cmd.extend(['-c', str(self.params['C'])])
            if 'gamma' in self.params:
                if self.params['gamma'] == 'scale':
                    # 计算 1/(n_features * X.var())
                    gamma_val = 1.0 / (X.shape[1] * np.var(X))
                    cmd.extend(['-g', str(gamma_val)])
                else:
                    cmd.extend(['-g', str(self.params['gamma'])])
            if 'kernel' in self.params:
                kernel_map = {'linear': 0, 'poly': 1, 'rbf': 2, 'sigmoid': 3}
                cmd.extend(['-t', str(kernel_map.get(self.params['kernel'], 2))])
            
            # 添加概率估计
            if self.params.get('probability', False):
                cmd.append('-b')
                cmd.append('1')
            
            # 添加输出文件
            model_path = train_path + '.model'
            cmd.extend([train_path, model_path])
            
            if verbose:
                print(f"执行命令: {' '.join(cmd)}")
                print("开始训练...")
                print("-" * 50)
            
            # 执行训练，实时显示输出
            if verbose:
                # 实时显示训练过程
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                        text=True, bufsize=1, universal_newlines=True)
                
                output_lines = []
                classifier_count = 0
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        print(line)
                        output_lines.append(line)
                        
                        # 检测到新的分类器训练完成
                        if 'optimization finished' in line:
                            classifier_count += 1
                            print(f"第 {classifier_count} 个分类器训练完成")
                
                # 等待进程完成
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"LIBSVM 训练失败，返回码: {return_code}")
                
                # 解析输出
                full_output = '\n'.join(output_lines)
                self._parse_libsvm_output(full_output)
            else:
                # 不显示详细输出
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode != 0:
                    raise RuntimeError(f"LIBSVM 训练失败: {result.stderr}")
                self._parse_libsvm_output(result.stdout)
            self.model_path = model_path
            
        finally:
            # 清理临时文件
            if os.path.exists(train_path):
                os.unlink(train_path)
        
        end_time = time.time()
        self.training_history['training_time'] = end_time - start_time
        
        if verbose:
            print(f"训练完成！耗时: {self.training_history['training_time']:.2f} 秒")
    
    def predict(self, X):
        """预测"""
        if self.model_path is None:
            raise ValueError("模型未训练")
        
        # 创建测试数据文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.libsvm', delete=False) as test_file:
            test_path = test_file.name
        
        try:
            # 转换测试数据
            self._create_libsvm_data(X, np.zeros(len(X)), test_path)
            
            # 预测
            output_path = test_path + '.out'
            cmd = ['svm-predict', test_path, self.model_path, output_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"预测失败: {result.stderr}")
            
            # 读取预测结果
            with open(output_path, 'r') as f:
                predictions = [int(line.strip()) for line in f.readlines()]
            
            return np.array(predictions)
            
        finally:
            # 清理临时文件
            for path in [test_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model_path is None:
            raise ValueError("模型未训练")
        
        # 创建测试数据文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.libsvm', delete=False) as test_file:
            test_path = test_file.name
        
        try:
            # 转换测试数据
            self._create_libsvm_data(X, np.zeros(len(X)), test_path)
            
            # 预测概率
            output_path = test_path + '.out'
            cmd = ['svm-predict', '-b', '1', test_path, self.model_path, output_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"概率预测失败: {result.stderr}")
            
            # 读取概率结果
            probabilities = []
            with open(output_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        # LIBSVM 概率输出格式: label prob1 prob2 ...
                        parts = line.strip().split()
                        if len(parts) > 1:
                            probs = [float(p) for p in parts[1:]]
                            probabilities.append(probs)
            
            return np.array(probabilities)
            
        finally:
            # 清理临时文件
            for path in [test_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def save(self, model_path):
        """保存模型"""
        if self.model_path is None:
            raise ValueError("模型未训练")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # 复制模型文件
        import shutil
        shutil.copy2(self.model_path, model_path)
    
    def load(self, model_path):
        """加载模型"""
        self.model_path = model_path
    
    def get_training_info(self):
        """获取训练信息"""
        return self.training_history
