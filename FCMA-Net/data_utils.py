import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil

class DataPreprocessor:
    
    def __init__(self, config):
        self.config = config
        self.image_size = config.IMAGE_SIZE
        
    def resize_image(self, image_path, output_path=None):
        if output_path is None:
            output_path = image_path
            
        image = cv2.imread(image_path)
        if image is None:
            return False
            
        resized = cv2.resize(image, (self.image_size, self.image_size))
        cv2.imwrite(output_path, resized)
        return True
    
    def enhance_image(self, image_path, output_path=None):
        if output_path is None:
            output_path = image_path
            
        image = Image.open(image_path)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        image = image.filter(ImageFilter.SHARPEN)
        image.save(output_path)
        return True
    
    def create_validation_split(self, train_ratio=0.8, random_state=42):
        train_dir = self.config.TRAIN_DIR
        val_dir = self.config.VAL_DIR
        
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
            os.makedirs(os.path.join(val_dir, 'real'))
            os.makedirs(os.path.join(val_dir, 'fake'))
        
        real_train_dir = os.path.join(train_dir, 'real')
        real_val_dir = os.path.join(val_dir, 'real')
        
        real_images = [f for f in os.listdir(real_train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        real_train, real_val = train_test_split(
            real_images, train_size=train_ratio, random_state=random_state
        )
        
        fake_train_dir = os.path.join(train_dir, 'fake')
        fake_val_dir = os.path.join(val_dir, 'fake')
        
        fake_images = [f for f in os.listdir(fake_train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        fake_train, fake_val = train_test_split(
            fake_images, train_size=train_ratio, random_state=random_state
        )
        
        for img in real_val:
            src = os.path.join(real_train_dir, img)
            dst = os.path.join(real_val_dir, img)
            shutil.move(src, dst)
            
        for img in fake_val:
            src = os.path.join(fake_train_dir, img)
            dst = os.path.join(fake_val_dir, img)
            shutil.move(src, dst)
        
        print(f"Train real images: {len(real_train)}")
        print(f"Val real images: {len(real_val)}")
        print(f"Train fake images: {len(fake_train)}")
        print(f"Val fake images: {len(fake_val)}")
        
        return {
            'real_train': real_train,
            'real_val': real_val,
            'fake_train': fake_train,
            'fake_val': fake_val
        }
    
    def get_augmentation_transforms(self):
        train_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        
        val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        
        return train_transform, val_transform
    
    def get_torchvision_transforms(self):
        from torchvision import transforms
        
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return train_transform, val_transform
    
    def analyze_dataset(self):
        train_dir = self.config.TRAIN_DIR
        val_dir = self.config.VAL_DIR
        
        stats = {}
        
        if os.path.exists(train_dir):
            real_train = len([f for f in os.listdir(os.path.join(train_dir, 'real')) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            fake_train = len([f for f in os.listdir(os.path.join(train_dir, 'fake')) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            stats['train'] = {'real': real_train, 'fake': fake_train, 'total': real_train + fake_train}
        
        if os.path.exists(val_dir):
            real_val = len([f for f in os.listdir(os.path.join(val_dir, 'real')) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            fake_val = len([f for f in os.listdir(os.path.join(val_dir, 'fake')) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            stats['val'] = {'real': real_val, 'fake': fake_val, 'total': real_val + fake_val}
        
        return stats
    
    def plot_dataset_distribution(self, stats):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if 'train' in stats:
            train_data = stats['train']
            labels = ['Real', 'Fake']
            sizes = [train_data['real'], train_data['fake']]
            colors = ['lightblue', 'lightcoral']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Train Set Distribution (Total: {train_data["total"]})')
        
        if 'val' in stats:
            val_data = stats['val']
            labels = ['Real', 'Fake']
            sizes = [val_data['real'], val_data['fake']]
            colors = ['lightgreen', 'lightpink']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Val Set Distribution (Total: {val_data["total"]})')
        
        plt.tight_layout()
        plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def check_image_quality(self, image_dir, sample_size=100):
        import random
        
        all_images = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, file))
        
        if len(all_images) == 0:
            print("No images found")
            return
        
        sample_images = random.sample(all_images, min(sample_size, len(all_images)))
        
        quality_info = []
        for img_path in sample_images:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    quality_info.append({
                        'path': img_path,
                        'size': (width, height),
                        'aspect_ratio': width / height,
                        'file_size': os.path.getsize(img_path) / 1024
                    })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if quality_info:
            print(f"Image quality analysis (samples: {len(quality_info)})")
            print("-" * 50)
            
            sizes = [info['size'] for info in quality_info]
            aspect_ratios = [info['aspect_ratio'] for info in quality_info]
            file_sizes = [info['file_size'] for info in quality_info]
            
            print(f"Avg size: {np.mean(sizes, axis=0)}")
            print(f"Avg aspect ratio: {np.mean(aspect_ratios):.2f}")
            print(f"Avg file size: {np.mean(file_sizes):.1f} KB")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            widths = [size[0] for size in sizes]
            heights = [size[1] for size in sizes]
            axes[0].scatter(widths, heights, alpha=0.6)
            axes[0].set_xlabel('Width')
            axes[0].set_ylabel('Height')
            axes[0].set_title('Image Size Distribution')
            
            axes[1].hist(aspect_ratios, bins=20, alpha=0.7)
            axes[1].set_xlabel('Aspect Ratio')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Aspect Ratio Distribution')
            
            axes[2].hist(file_sizes, bins=20, alpha=0.7)
            axes[2].set_xlabel('File Size (KB)')
            axes[2].set_ylabel('Count')
            axes[2].set_title('File Size Distribution')
            
            plt.tight_layout()
            plt.savefig('image_quality_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    from config import config
    
    preprocessor = DataPreprocessor(config)
    
    print("Analyzing dataset...")
    stats = preprocessor.analyze_dataset()
    print(stats)
    
    preprocessor.plot_dataset_distribution(stats)
    
    print("\nChecking image quality...")
    preprocessor.check_image_quality(config.DATA_ROOT)
    
    print("\nCreate validation split? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("Creating validation split...")
        split_info = preprocessor.create_validation_split()
        print("Validation split complete!")
        
        stats = preprocessor.analyze_dataset()
        preprocessor.plot_dataset_distribution(stats)

if __name__ == "__main__":
    main()
