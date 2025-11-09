import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor
import cv2
import os

class ViTFaceForgeryDetector(nn.Module):
    
    def __init__(self, num_classes=2, model_name='facebook/dino-v2-vitb14'):
        super(ViTFaceForgeryDetector, self).__init__()
        
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        cls_output = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

def load_model(model_path, device):
    model = ViTFaceForgeryDetector(num_classes=2)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded, val acc: {checkpoint['val_acc']:.4f}")
    return model

def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image

def predict_single_image(model, image_tensor, device):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        
        _, predicted_class = torch.max(outputs, 1)
        confidence = probabilities[0][predicted_class].item()
        
        real_prob = probabilities[0][0].item()
        fake_prob = probabilities[0][1].item()
        
        return predicted_class.item(), confidence, real_prob, fake_prob

def visualize_prediction(image, prediction, confidence, real_prob, fake_prob, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
    ax1.axis('off')
    
    labels = ['Real', 'Fake']
    probs = [real_prob, fake_prob]
    colors = ['green' if prediction == 0 else 'red', 'red' if prediction == 1 else 'green']
    
    bars = ax2.bar(labels, probs, color=colors, alpha=0.7)
    ax2.set_ylabel('Probability')
    ax2.set_title('Classification Probability')
    ax2.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def batch_predict(model, image_dir, device, output_dir='predictions'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(image_dir, filename)
            
            try:
                image_tensor, image = preprocess_image(image_path)
                
                prediction, confidence, real_prob, fake_prob = predict_single_image(
                    model, image_tensor, device
                )
                
                result = {
                    'filename': filename,
                    'prediction': 'Fake' if prediction == 1 else 'Real',
                    'confidence': confidence,
                    'real_prob': real_prob,
                    'fake_prob': fake_prob
                }
                results.append(result)
                
                output_path = os.path.join(output_dir, f"pred_{filename}")
                visualize_prediction(image, prediction, confidence, real_prob, fake_prob, output_path)
                
                print(f"{filename}: {result['prediction']} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_path = 'best_face_forgery_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found {model_path}")
        print("Please run training script first")
        return
    
    model = load_model(model_path, device)
    
    while True:
        print("\n" + "="*50)
        print("Face Forgery Detection System")
        print("="*50)
        print("1. Detect single image")
        print("2. Batch detect images")
        print("3. Exit")
        
        choice = input("Select operation (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            
            if not os.path.exists(image_path):
                print("Error: Image file not found")
                continue
            
            try:
                image_tensor, image = preprocess_image(image_path)
                
                prediction, confidence, real_prob, fake_prob = predict_single_image(
                    model, image_tensor, device
                )
                
                print(f"\nPrediction: {'Fake' if prediction == 1 else 'Real'}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Real prob: {real_prob:.3f}")
                print(f"Fake prob: {fake_prob:.3f}")
                
                visualize_prediction(image, prediction, confidence, real_prob, fake_prob)
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
        
        elif choice == '2':
            image_dir = input("Enter image directory: ").strip()
            
            if not os.path.exists(image_dir):
                print("Error: Directory not found")
                continue
            
            try:
                results = batch_predict(model, image_dir, device)
                
                import json
                with open('batch_predictions.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"\nBatch detection complete, processed {len(results)} images")
                print("Results saved to batch_predictions.json")
                
            except Exception as e:
                print(f"Error in batch detection: {str(e)}")
        
        elif choice == '3':
            print("Thanks for using!")
            break
        
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()
