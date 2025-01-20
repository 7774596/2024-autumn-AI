import os, torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms, models
from transformers import BertTokenizer
from torch.utils.data import Dataset
import albumentations as A
from torchvision.models import ResNet50_Weights

image_model = models.resnet50(weights=ResNet50_Weights.DEFAULT) # ResNet50提取图像特征
image_model = torch.nn.Sequential(*list(image_model.children())[:-1])
image_model.eval()

train_transform = A.Compose([ #图片数据增强
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.ElasticTransform(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(p=0.5),
        A.Emboss(p=0.5),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class MultimodalDataset(Dataset):
    def __init__(self, texts, images, labels, tokenizer, max_length=128):
        self.texts = texts
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image_features = self.images[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image_features': image_features,
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultimodalDataLoader:
    def __init__(self, data_dir, text_model_name='bert-base-uncased'):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
        
    def load_data(self, df, sample_ratio=1.0, is_train=True):
        texts, images, labels = [], [], []
        df = df.sample(frac=sample_ratio, random_state=42)
        transform = train_transform if is_train else val_transform
        
        for guid, label in df.values:
            guid = str(guid).strip()
            text_path = os.path.join(self.data_dir, f'{guid}.txt')
            image_path = os.path.join(self.data_dir, f'{guid}.jpg')

            if not os.path.exists(text_path) or not os.path.exists(image_path):
                continue

            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(text_path, 'r', encoding='iso-8859-1') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    continue

            try:
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)  
                image = transform(image=image)['image']  
                image = torch.from_numpy(image).permute(2, 0, 1).float()  #转换为张量
            except Exception as e:
                continue

            with torch.no_grad():
                image_features = image_model(image.unsqueeze(0))  
                image_features = image_features.view(-1) 

            texts.append(text)
            images.append(image_features)
            labels.append(label)

        labels = np.array(labels)
        return texts, images, labels

    def encode_texts(self, texts, max_length=512): #文本格式的编码
        return self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )