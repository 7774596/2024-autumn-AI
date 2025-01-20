import torch, os
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, mode='multimodal', fold=0):
    best_val_loss = float('inf')
    patience = 4  #早停忍耐度4
    patience_counter = 0

    warmup_epochs = 2 #预热轮次2
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    scheduler_cosine = get_cosine_schedule_with_warmup( #余弦退火学习率调度器
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for input_ids, attention_mask, image_features, labels in train_progress:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image_features, mode=mode)
            
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            scheduler_cosine.step() 
            
            train_loss += loss.item()
            train_progress.set_postfix({'Train Loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, mode=mode)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            if mode == 'multimodal':
                torch.save(model.state_dict(), 'models/best_multimodal_model.pth')  #只保存最佳的Multi模型
                print(f"Best multimodal model saved with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses, val_accuracies

def validate_model(model, val_loader, criterion, mode='multimodal'):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, image_features, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask, image_features, mode=mode)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_accuracy += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader.dataset)
    
    return avg_val_loss, avg_val_accuracy

def plot_training_curves(train_losses, val_losses, val_accuracies, mode): #绘制loss曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{mode} - Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title(f'{mode} - Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{mode}_training_curves.png')
    plt.close()  