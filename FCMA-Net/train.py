import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import seaborn as sns
import json
from datetime import datetime
from FTMA import Ftma
from freq_branch import HaarDWT, FreqBranch
from SFCM import Sfcm
import copy
from config import config

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * (1 - targets) + (1 - self.alpha) * targets
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FaceForgeryDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.methods = []
        
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)
                    self.methods.append('real')
        
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)
                    method = self.extract_method_from_filename(img_name)
                    self.methods.append(method)
    
    def extract_method_from_filename(self, filename):
        import re
        if filename.startswith('fake_'):
            match = re.match(r'fake_([^_]+)_', filename)
            if match:
                return match.group(1)
        return 'unknown'
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        method = self.methods[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, method

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomApply([
            transforms.Resize(int(image_size*0.6)),
            transforms.Resize(image_size)
        ], p=0.25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.3, 3.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class ViTWithFreqFusion(nn.Module):
    
    def __init__(self, num_classes=2, model_name='facebook/dino-v2-vitb14', 
                 fuse_type='concat', freq_dim=256, reduction_ratio=16, use_aux_head=False,
                 use_cross_attention=True):
        super(ViTWithFreqFusion, self).__init__()
        
        model_path = './pretrained_models/dino-v2-vitb14/pytorch_model.bin'
        if os.path.exists(model_path):
            import torch.hub
            self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            self.vit.load_state_dict(state_dict)
        else:
            import torch.hub
            self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        self.hidden_size = 768
        self.freq_dim = freq_dim
        self.fuse_type = fuse_type
        self.use_aux_head = use_aux_head
        self.use_cross_attention = use_cross_attention
        
        self.token_se = Ftma(channels=self.hidden_size, reduction_ratio=reduction_ratio)
        self.se_alpha = nn.Parameter(torch.tensor(1e-3))
        
        self.dwt = HaarDWT()
        if use_cross_attention:
            self.freq_branch = FreqBranch(in_ch=12, embed_dim=self.hidden_size, output_tokens=True)
            self.cross_attention = Sfcm(hidden_dim=self.hidden_size, num_heads=8)
            self.freq_proj = nn.Linear(self.hidden_size, freq_dim)
        else:
            self.freq_branch = FreqBranch(in_ch=12, embed_dim=freq_dim, output_tokens=False)
        
        if use_cross_attention:
            fuse_input_dim = self.hidden_size + freq_dim
            self.fuse_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(fuse_input_dim, 256),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
        else:
            if fuse_type == 'concat':
                fuse_input_dim = self.hidden_size + freq_dim
                self.fuse_head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(fuse_input_dim, 256),
                    nn.GELU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, num_classes)
                )
            elif fuse_type == 'residual':
                self.gamma = nn.Parameter(torch.tensor(1e-3))
                self.freq_proj = nn.Linear(freq_dim, self.hidden_size)
                self.fuse_head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.hidden_size, 256),
                    nn.GELU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, num_classes)
                )
            else:
                raise ValueError(f"Unsupported fusion type: {fuse_type}")
        
        if use_aux_head:
            self.aux_head = nn.Linear(freq_dim, num_classes)
    
    def forward(self, pixel_values, return_aux=False):
        features = self.vit.forward_features(pixel_values)
        
        if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
            cls_token = features["x_norm_clstoken"].unsqueeze(1)
            patch_tokens = features["x_norm_patchtokens"]
        else:
            all_tokens = features["x_prenorm"]
            cls_token = all_tokens[:, 0:1, :]
            num_register_tokens = getattr(self.vit, 'num_register_tokens', 0)
            if num_register_tokens > 0:
                patch_tokens = all_tokens[:, 1+num_register_tokens:, :]
            else:
                patch_tokens = all_tokens[:, 1:, :]
        
        patch_tokens_attended = self.token_se(patch_tokens)
        
        if self.use_cross_attention:
            mean = pixel_values.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = pixel_values.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            pixel_values_rgb = pixel_values * std + mean
            
            dwt_output = self.dwt(pixel_values_rgb)
            freq_tokens = self.freq_branch(dwt_output)
            
            enhanced_patch_tokens, enhanced_freq_tokens = self.cross_attention(
                patch_tokens_attended, freq_tokens
            )
            
            patch_summary = enhanced_patch_tokens.mean(dim=1, keepdim=True)
            alpha_weighted = torch.sigmoid(self.se_alpha)
            cls_refined = cls_token + alpha_weighted * patch_summary
            cls_feat = cls_refined.squeeze(1)
            
            freq_feat = enhanced_freq_tokens.mean(dim=1)
            freq_feat = self.freq_proj(freq_feat)
            
            fused_feat = torch.cat([cls_feat, freq_feat], dim=1)
            logits = self.fuse_head(fused_feat)
            
        else:
            patch_summary = patch_tokens_attended.mean(dim=1, keepdim=True)
            alpha_weighted = torch.sigmoid(self.se_alpha)
            cls_refined = cls_token + alpha_weighted * patch_summary
            cls_feat = cls_refined.squeeze(1)
            
            dwt_output = self.dwt(pixel_values)
            freq_feat = self.freq_branch(dwt_output)
            
            if self.fuse_type == 'concat':
                fused_feat = torch.cat([cls_feat, freq_feat], dim=1)
                logits = self.fuse_head(fused_feat)
            elif self.fuse_type == 'residual':
                freq_proj = self.freq_proj(freq_feat)
                enhanced_cls = cls_feat + self.gamma * freq_proj
                logits = self.fuse_head(enhanced_cls)
        
        if return_aux and self.use_aux_head:
            if self.use_cross_attention:
                freq_feat = enhanced_freq_tokens.mean(dim=1)
                aux_logits = self.aux_head(freq_feat)
            else:
                aux_logits = self.aux_head(freq_feat)
            return logits, aux_logits
        else:
            return logits


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, ema=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, labels, methods) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Batch': f'{batch_idx+1}/{len(dataloader)}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_methods = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        
        for images, labels, methods in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_methods.extend(methods)
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5
    
    real_indices = [i for i, label in enumerate(all_labels) if label == 0]
    fake_indices = [i for i, label in enumerate(all_labels) if label == 1]
    
    real_acc = 0.0
    fake_acc = 0.0
    
    if len(real_indices) > 0:
        real_preds = [all_preds[i] for i in real_indices]
        real_labels = [all_labels[i] for i in real_indices]
        real_acc = accuracy_score(real_labels, real_preds)
    
    if len(fake_indices) > 0:
        fake_preds = [all_preds[i] for i in fake_indices]
        fake_labels = [all_labels[i] for i in fake_indices]
        fake_acc = accuracy_score(fake_labels, fake_preds)
    
    method_metrics = {}
    fake_methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    for method in fake_methods:
        method_indices = [i for i, m in enumerate(all_methods) if m == method]
        if len(method_indices) > 0:
            method_preds = [all_preds[i] for i in method_indices]
            method_labels = [all_labels[i] for i in method_indices]
            method_probs = [all_probs[i] for i in method_indices]
            
            method_acc = accuracy_score(method_labels, method_preds)
            
            real_indices = [i for i, label in enumerate(all_labels) if label == 0]
            if len(real_indices) > 0:
                combined_indices = real_indices + method_indices
                combined_labels = [all_labels[i] for i in combined_indices]
                combined_probs = [all_probs[i] for i in combined_indices]
                
                try:
                    method_auc = roc_auc_score(combined_labels, combined_probs)
                except ValueError:
                    method_auc = 0.5
            else:
                method_auc = 0.5
            
            method_metrics[method] = {
                'acc': method_acc,
                'auc': method_auc,
                'count': len(method_indices)
            }
        else:
            method_metrics[method] = {
                'acc': 0.0,
                'auc': 0.5,
                'count': 0
            }
    
    return epoch_loss, epoch_acc, epoch_auc, all_preds, all_labels, real_acc, fake_acc, method_metrics

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd, esd = model.state_dict(), self.ema.state_dict()
        for k in esd.keys():
            esd[k].copy_(esd[k]*d + msd[k]*(1.0-d))

def plot_training_history(history):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Train and Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Train and Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(history['val_auc'], label='Val AUC', color='green')
    ax3.set_title('Val AUC')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(history['real_acc'], label='Real Acc', color='blue')
    ax4.plot(history['fake_acc'], label='Fake Acc', color='red')
    ax4.set_title('Real vs Fake Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 50
    IMAGE_SIZE = 224
    
    train_dir = config.TRAIN_DIR
    val_dir = config.VAL_DIR
    
    train_transform, val_transform = get_transforms(IMAGE_SIZE)
    
    train_dataset = FaceForgeryDataset(train_dir, transform=train_transform)
    val_dataset = FaceForgeryDataset(val_dir, transform=val_transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = ViTWithFreqFusion(num_classes=2, model_name='facebook/dino-v2-vitb14',
                              fuse_type='concat', freq_dim=256, reduction_ratio=16, use_aux_head=False,
                              use_cross_attention=True)
    model = model.to(device)
    
    start_epoch = 0
    checkpoint = None
    if os.path.exists('best_face_forgery_model.pth'):
        print("Loading saved model...")
        try:
            checkpoint = torch.load('best_face_forgery_model.pth', map_location=device, weights_only=False)
            
            if 'ema_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['ema_state_dict'], strict=True)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Model loaded, resume from Epoch {start_epoch}, best val acc: {checkpoint['val_acc']:.4f}")
        except Exception as e:
            print(f"Failed to load model: {e}, training from scratch")
            start_epoch = 0
            checkpoint = None
    else:
        print("Training from scratch")
    
    real_cnt = len(os.listdir(os.path.join(train_dir, 'real'))) if os.path.exists(os.path.join(train_dir, 'real')) else 1
    fake_cnt = len(os.listdir(os.path.join(train_dir, 'fake'))) if os.path.exists(os.path.join(train_dir, 'fake')) else 1
    
    print(f"Train set distribution - Real: {real_cnt}, Fake: {fake_cnt}")
    
    criterion = FocalLoss(alpha=0.6, gamma=2.0)

    backbone, head = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ["token_se", "se_alpha", "fuse_head", "freq", "aux_head", "freq_proj", "gamma", "cross_attention"]):
            head.append(p)
        elif "vit" in n:
            backbone.append(p)
    
    optimizer = optim.AdamW([
        {"params": backbone, "lr": 6e-6, "weight_decay": 0.05},
        {"params": head, "lr": 6e-5, "weight_decay": 0.0},
    ], betas=(0.9, 0.999), eps=1e-8)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))
    def lr_lambda(step):
        import numpy as np
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * t))
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'real_acc': [],
        'fake_acc': []
    }
    
    best_val_acc = 0.0
    patience = 3
    no_improve = 0
    ema = ModelEMA(model, decay=0.999)
    
    if start_epoch > 0 and checkpoint is not None:
        try:
            history = checkpoint['history']
            best_val_acc = checkpoint['val_acc']
            if 'ema_state_dict' in checkpoint:
                ema.ema.load_state_dict(checkpoint['ema_state_dict'], strict=True)
        except Exception as e:
            print(f"Failed to restore history: {e}, starting fresh")
    
    print("Start training...")
    if start_epoch > 0:
        print(f"Resume from Epoch {start_epoch}, remaining {NUM_EPOCHS - start_epoch} epochs")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, ema)
        val_loss, val_acc, val_auc, val_preds, val_labels, real_acc, fake_acc, method_metrics = validate_epoch(ema.ema, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['real_acc'].append(real_acc)
        history['fake_acc'].append(fake_acc)
        
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Real acc: {real_acc:.4f}, Fake acc: {fake_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
                'history': history
            }, 'best_face_forgery_model.pth')
            print(f"Saved best model, Val acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{patience}")
    
    print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")
    
    plot_training_history(history)
    plot_confusion_matrix(val_labels, val_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
    print(f"\nFinal evaluation:")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {val_auc:.4f}")
    
    with open('training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()