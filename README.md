# 2024秋当代人工智能课程 实验五 多模态情感分析任务

学号：10212140414 姓名：盛子骜

## **项目简介**

本项目旨在通过结合文本和图像信息，实现多模态情感分析。项目使用了预训练的 BERT 模型提取文本特征，以及预训练的 ResNet50 模型提取图像特征。通过多头注意力机制融合文本和图像特征，最终使用全连接层进行分类。项目支持三种模式：仅文本（`text_only`）、仅图像（`image_only`）和多模态融合（`multimodal`），并通过消融实验验证了多模态融合的有效性。

## **项目参考**
本代码主要参考了以下库实现：

- 深度学习框架：torch（PyTorch）
- 自然语言处理：transformers（Hugging Face）
- 图像处理：PIL（Pillow）、albumentations
- 数据处理与工具：pandas、numpy、scikit-learn
- 进度条与日志：tqdm
- 其他工具：os、matplotlib
---

## **项目结构**

```
/
├── lab5-data/               # 数据目录
│   ├── train.txt            # 训练集
│   ├── test_without_label.txt # 测试集
|   ├── test_predictions.txt # 生成的预测文件
│   └── data/                # 图像和文本真实数据
├── model/                   # 模型保存目录
│   └── best_multimodal_model.pth # 保存的最佳Multimodel模型
├── utils/                   # 工具函数
│   └── data_loader.py       # 数据加载器
├── train.py                 # 主训练脚本
├── predict.py               # 预测脚本
├── model.py                 # 模型定义
├── train_utils.py           # 训练函数和辅助函数
├── requirements.txt         # 依赖库
└── README.md                # 项目说明
```

---
## **依赖列表**

- albumentations==2.0.0
- matplotlib==3.10.0
- numpy==2.2.2
- pandas==2.2.3
- Pillow==11.1.0
- scikit_learn==1.6.1
- torch==2.5.1
- torchvision==0.20.1
- torchviz==0.0.3
- tqdm==4.66.5
- transformers==4.47.1

运行以下命令安装所需的Python库：
```bash
pip install -r requirements.txt
```

---


## **数据准备**

数据集请按照项目结构中的格式进行放置，由于数据集过于庞大就没有上传到仓库中。

## **模型训练**

运行以下命令训练模型：
```bash
python train.py
```

### **训练参数**
- `--data_dir`：数据目录路径（默认：`lab5-data/data`）。
- `--train_file`：训练集文件路径（默认：`lab5-data/train.txt`）。
- `--model_dir`：模型保存目录（默认：`model`）。
- `--num_classes`：情感类别数量（默认：3）。
- `--epochs`：训练轮数（默认：10）。
- `--n_splits`：训练折数（默认：3）。
- `--batch_size`：批大小（默认：8）。

### **训练结果**
- 训练过程中会输出每个epoch的训练损失、验证损失和验证准确率。
- 最佳模型会保存到 `models/best_multimodal_model.pth`。

---

## **模型预测**

运行以下命令对测试集进行预测：
```bash
python predict.py
```

### **预测参数**
- `--test_file`：测试集文件路径（默认：`lab5-data/test_without_label.txt`）。
- `--data_dir`：数据目录路径（默认：`lab5-data/data`）。
- `--model_path`：模型路径（默认：`model/best_multimodal_model.pth`）。
- `--output_file`：预测结果保存路径（默认：`lab5-data/test_predictions.txt`）。

### **预测结果**
- 预测结果会保存到 `output_file`，包含两列：`guid` 和 `tag`。
- 示例：
  ```
  guid,tag
  12345,positive
  67890,negative
  ```


