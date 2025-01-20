import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from utils.data_loader import MultimodalDataLoader
from train import ImprovedMultimodalModel  
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_best_model(model_path, text_feature_dim=768, image_feature_dim=2048, num_classes=3):
    model = ImprovedMultimodalModel(
        text_feature_dim=text_feature_dim,
        image_feature_dim=image_feature_dim,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  
    return model


def load_test_data(test_file, data_dir):
    test_df = pd.read_csv(test_file, sep=',', header=0, names=['guid', 'tag'], dtype={'guid': str})
    data_loader = MultimodalDataLoader(data_dir)
    texts, image_features, _ = data_loader.load_data(test_df)
    return texts, image_features, test_df['guid'].tolist()


def preprocess_test_data(texts, image_features, tokenizer):
    encodings = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    image_features = torch.stack(image_features)
    return encodings['input_ids'], encodings['attention_mask'], image_features


def predict(model, input_ids, attention_mask, image_features, batch_size=16):
    dataset = TensorDataset(input_ids, attention_mask, image_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for input_ids_batch, attention_mask_batch, image_features_batch in tqdm(dataloader, desc="Predicting"):
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            image_features_batch = image_features_batch.to(device)

            outputs = model(input_ids_batch, attention_mask_batch, image_features_batch)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds

def convert_labels(preds, label_mapping):
    return [label_mapping[pred] for pred in preds]

def save_predictions(guids, preds, output_file):
    pred_df = pd.DataFrame({'guid': guids, 'tag': preds})
    pred_df.to_csv(output_file, index=False, header=True)
    print(f"Predictions saved to {output_file}")


def main():
    test_file = 'lab5-data/test_without_label.txt'  
    data_dir = 'lab5-data/data'  
    model_path = 'models/best_multimodal_model.pth' 
    output_file = 'lab5-data/test_predictions.txt' #保存的预测文件

    label_mapping = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }

    print("Loading the best model...")
    model = load_best_model(model_path, num_classes=3) 

    print("Loading test data...")
    texts, image_features, guids = load_test_data(test_file, data_dir)

    print("Preprocessing test data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_mask, image_features = preprocess_test_data(texts, image_features, tokenizer)

    print("Making predictions...")
    preds = predict(model, input_ids, attention_mask, image_features)

    print("Converting labels...")
    pred_tags = convert_labels(preds, label_mapping)

    print("Saving predictions...")
    save_predictions(guids, pred_tags, output_file)

if __name__ == '__main__':
    main()