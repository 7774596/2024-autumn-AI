import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from model import ImprovedMultimodalModel
from train_utils import train_model, plot_training_curves
from sklearn.utils.class_weight import compute_class_weight
from utils.data_loader import MultimodalDataLoader
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    train_df = pd.read_csv('lab5-data/train.txt', sep=',', header=0, names=['guid', 'tags'], dtype={'guid': str}) 
    data_loader = MultimodalDataLoader('lab5-data/data')
    texts, image_features, labels = data_loader.load_data(train_df, is_train=True) #从train.df中获取的guid找data中的真实数据加载

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels_encoded),
        y=labels_encoded
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    modes = ['text_only', 'image_only', 'multimodal'] #模式列表，三种模式都跑一遍
    results = {mode: {'train_losses': [], 'val_losses': [], 'val_accuracies': []} for mode in modes}

    for mode in modes:
        print(f'\nRunning ablation study for {mode} mode...')
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels_encoded)):
            print(f'\nFold {fold+1}/{n_splits}')
            
            X_train_text = [texts[i] for i in train_idx] #划分这个fold的数据
            X_val_text = [texts[i] for i in val_idx]
            
            X_train_image = torch.stack([image_features[i] for i in train_idx])
            X_val_image = torch.stack([image_features[i] for i in val_idx])
        
            y_train = labels_encoded[train_idx]
            y_val = labels_encoded[val_idx]

            train_encodings = data_loader.encode_texts(X_train_text) #文本数据编码
            val_encodings = data_loader.encode_texts(X_val_text)


            train_dataset = TensorDataset(
                train_encodings['input_ids'], 
                train_encodings['attention_mask'],
                X_train_image, 
                torch.tensor(y_train)
            )
            val_dataset = TensorDataset(
                val_encodings['input_ids'], 
                val_encodings['attention_mask'],
                X_val_image, 
                torch.tensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)

            model = ImprovedMultimodalModel(
                text_feature_dim=768, 
                image_feature_dim=2048, 
                num_classes=len(np.unique(labels_encoded))
            ).to(device)
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001) #优化器设置

            train_losses, val_losses, val_accuracies = train_model( #训练
                model, train_loader, val_loader, criterion, optimizer, epochs=10, mode=mode, fold=fold
            )
            
            results[mode]['train_losses'].append(train_losses)
            results[mode]['val_losses'].append(val_losses)
            results[mode]['val_accuracies'].append(val_accuracies)
        
        avg_val_accuracy = np.mean([acc[-1] for acc in results[mode]['val_accuracies']])
        print(f'\n{mode} mode - Average validation accuracy: {avg_val_accuracy:.4f}')

    for mode in modes: #可视化每种模式的训练曲线

        min_length = min(len(train_losses) for train_losses in results[mode]['train_losses'])
        aligned_train_losses = [train_losses[:min_length] for train_losses in results[mode]['train_losses']]
        avg_train_losses = np.mean(aligned_train_losses, axis=0)

        min_length = min(len(val_losses) for val_losses in results[mode]['val_losses'])
        aligned_val_losses = [val_losses[:min_length] for val_losses in results[mode]['val_losses']]
        avg_val_losses = np.mean(aligned_val_losses, axis=0)

        min_length = min(len(val_accuracies) for val_accuracies in results[mode]['val_accuracies'])
        aligned_val_accuracies = [val_accuracies[:min_length] for val_accuracies in results[mode]['val_accuracies']]
        avg_val_accuracies = np.mean(aligned_val_accuracies, axis=0)

        plot_training_curves(avg_train_losses, avg_val_losses, avg_val_accuracies, mode)

    print("\nAblation study results:")
    for mode in modes:
        avg_val_accuracy = np.mean([acc[-1] for acc in results[mode]['val_accuracies']])
        print(f'{mode} mode - Average validation accuracy: {avg_val_accuracy:.4f}')

if __name__ == '__main__':
    main()