import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class ImprovedMultimodalModel(nn.Module):
    def __init__(self, text_feature_dim, image_feature_dim, num_classes):
        super(ImprovedMultimodalModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.image_feature_extractor = nn.Sequential( #图像特征形状变换 2048→128
            nn.Linear(image_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.text_feature_transformer = nn.Sequential( #文本特征形状变换 768→128
            nn.Linear(text_feature_dim, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.multihead_attention = nn.MultiheadAttention( #引入注意力机制
            embed_dim=128,  
            num_heads=8,    
            dropout=0.1     
        )
        
        self.fc_fusion1 = nn.Linear(128, 128)  
        self.fc_fusion2 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(128)
        
    def forward(self, input_ids, attention_mask, image_features, mode='multimodal'):# 根据不同特征提取模式选择前向算法
        if mode == 'text_only':
            text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            text_features = self.text_feature_transformer(text_features)
            output = self.classifier(text_features)
            return output
        
        elif mode == 'image_only':
            image_features = self.image_feature_extractor(image_features.view(image_features.size(0), -1))
            output = self.classifier(image_features)
            return output
        
        elif mode == 'multimodal':
            text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            text_features = self.text_feature_transformer(text_features)
            text_features = text_features.unsqueeze(1) 

            image_features = self.image_feature_extractor(image_features.view(image_features.size(0), -1))
            image_features = image_features.unsqueeze(1)  
            
            combined_features = torch.cat((text_features, image_features), dim=1)  
            combined_features = combined_features.permute(1, 0, 2)  

            attn_output, _ = self.multihead_attention( #引入注意力机制
                combined_features,  
                combined_features, 
                combined_features   
            )

            attn_output = attn_output.permute(1, 0, 2)       
            fusion = attn_output.mean(dim=1)  #使用残差连接
            fusion = self.dropout(F.relu(self.fc_fusion1(fusion)))
            fusion_residual = self.fc_fusion2(fusion)
            fusion = self.layer_norm(fusion_residual + self.dropout(fusion))
            
            output = self.classifier(fusion)
            return output
        
        else:
            raise ValueError("Invalid mode. Choose from 'text_only', 'image_only', or 'multimodal'.")