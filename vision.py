import torch
from torchviz import make_dot
from model import ImprovedMultimodalModel

model = ImprovedMultimodalModel(
    text_feature_dim=768, 
    image_feature_dim=2048, 
    num_classes=3  
)

input_ids = torch.randint(0, 100, (1, 128))  
attention_mask = torch.ones((1, 128))  
image_features = torch.randn((1, 2048))  

model.eval()

outputs = model(input_ids, attention_mask, image_features, mode='multimodal')
dot = make_dot(outputs, params=dict(model.named_parameters()))
dot.format = 'png'  
dot.render('model_structure')  