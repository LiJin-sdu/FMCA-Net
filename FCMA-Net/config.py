import os

class Config:
    
    DATA_ROOT = r'D:\DataSet\FF++C23_32frames'
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    VAL_DIR = os.path.join(DATA_ROOT, 'valid')
    
    IMAGE_SIZE = 224
    CHANNELS = 3
    NUM_CLASSES = 2
    
    MODEL_NAME = 'facebook/dino-v2-vitb14'
    HIDDEN_SIZE = 768
    
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 0.01
    
    AUGMENTATION_CONFIG = {
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 10,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }
    }
    
    OPTIMIZER_CONFIG = {
        'type': 'AdamW',
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    
    SCHEDULER_CONFIG = {
        'type': 'CosineAnnealingLR',
        'T_max': NUM_EPOCHS,
        'eta_min': 1e-7
    }
    
    SAVE_CONFIG = {
        'save_dir': 'checkpoints',
        'save_freq': 5,
        'save_best': True,
        'save_last': True
    }
    
    LOG_CONFIG = {
        'log_dir': 'logs',
        'tensorboard': True,
        'log_freq': 100
    }
    
    VAL_CONFIG = {
        'val_freq': 1,
        'early_stopping_patience': 10,
        'early_stopping_delta': 0.001
    }
    
    DEVICE = 'auto'
    
    @classmethod
    def get_device(cls):
        if cls.DEVICE == 'auto':
            import torch
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(cls.DEVICE)
    
    @classmethod
    def create_dirs(cls):
        dirs = [
            cls.SAVE_CONFIG['save_dir'],
            cls.LOG_CONFIG['log_dir']
        ]
        
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created dir: {dir_path}")
    
    @classmethod
    def print_config(cls):
        print("=" * 50)
        print("Face Forgery Detection Config")
        print("=" * 50)
        
        print(f"Data root: {cls.DATA_ROOT}")
        print(f"Train dir: {cls.TRAIN_DIR}")
        print(f"Val dir: {cls.VAL_DIR}")
        print(f"Image size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Device: {cls.get_device()}")
        print("=" * 50)

config = Config()

