import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# Configuración de colores para clasificación
COLORS = ['negro', 'blanco', 'azul', 'rojo', 'verde', 'amarillo', 'gris', 'marron', 'rosa', 'violeta']

class ColorClassifier(nn.Module):
    """Red neuronal para clasificar colores de ropa"""
    def __init__(self, num_classes=len(COLORS)):
        super(ColorClassifier, self).__init__()
        # Usamos ResNet como backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        # Modificamos la última capa para nuestras clases
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes * 2)  # *2 para torso y piernas
        )
        
        # Separamos las salidas para torso y piernas
        self.torso_classifier = nn.Linear(num_classes * 2, num_classes)
        self.legs_classifier = nn.Linear(num_classes * 2, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        torso_logits = self.torso_classifier(features)
        legs_logits = self.legs_classifier(features)
        return torso_logits, legs_logits

class PersonColorDataset(Dataset):
    """Dataset personalizado para imágenes de personas con colores anotados"""
    def __init__(self, image_dir, annotations_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Cargar anotaciones
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(COLORS)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.image_dir, annotation['image'])
        
        # Cargar imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extraer región de la persona usando bbox
        bbox = annotation['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        person_crop = image[y:y+h, x:x+w]
        
        # Redimensionar
        person_crop = cv2.resize(person_crop, (224, 224))
        
        if self.transform:
            person_crop = self.transform(person_crop)
        
        # Codificar colores
        torso_color = self.label_encoder.transform([annotation['torso_color']])[0]
        legs_color = self.label_encoder.transform([annotation['legs_color']])[0]
        
        return person_crop, torso_color, legs_color

def create_annotations_template():
    """Crea un template de anotaciones para tus propios datos"""
    template = {
        "annotations": [
            {
                "image": "persona1.jpg",
                "bbox": [100, 50, 200, 400],  # [x, y, width, height]
                "torso_color": "azul",
                "legs_color": "negro"
            },
            {
                "image": "persona2.jpg", 
                "bbox": [150, 80, 180, 350],
                "torso_color": "rojo",
                "legs_color": "azul"
            }
        ]
    }
    
    with open('annotations_template.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    print("Template de anotaciones creado: annotations_template.json")
    print("Modifica este archivo con tus propias imágenes y anotaciones")

class YOLOColorSystem:
    """Sistema completo de detección YOLO + clasificación de colores"""
    def __init__(self, yolo_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Cargar modelo YOLO
        if yolo_model_path:
            self.yolo_model = YOLO(yolo_model_path)
        else:
            # Usar modelo preentrenado de YOLO
            self.yolo_model = YOLO('yolov8n.pt')  # Puedes usar yolov8s.pt, yolov8m.pt, etc.
        
        # Inicializar clasificador de colores
        self.color_classifier = ColorClassifier()
        self.color_classifier.to(self.device)
        
        # Transformaciones para el clasificador
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(COLORS)
    
    def train_color_classifier(self, train_dir, annotations_file, epochs=50, batch_size=16):
        """Entrena el clasificador de colores"""
        print("Iniciando entrenamiento del clasificador de colores...")
        
        # Dataset y DataLoader
        dataset = PersonColorDataset(train_dir, annotations_file, 
                                   transform=transforms.Compose([
                                       transforms.ToPILImage(),
                                       transforms.Resize((224, 224)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])
                                   ]))
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer y loss
        optimizer = optim.Adam(self.color_classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Entrenamiento
        self.color_classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            correct_torso = 0
            correct_legs = 0
            total_samples = 0
            
            for batch_idx, (images, torso_labels, legs_labels) in enumerate(dataloader):
                images = images.to(self.device)
                torso_labels = torso_labels.to(self.device)
                legs_labels = legs_labels.to(self.device)
                
                optimizer.zero_grad()
                
                torso_pred, legs_pred = self.color_classifier(images)
                
                loss_torso = criterion(torso_pred, torso_labels)
                loss_legs = criterion(legs_pred, legs_labels)
                loss = loss_torso + loss_legs
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calcular precisión
                _, torso_predicted = torch.max(torso_pred.data, 1)
                _, legs_predicted = torch.max(legs_pred.data, 1)
                
                correct_torso += (torso_predicted == torso_labels).sum().item()
                correct_legs += (legs_predicted == legs_labels).sum().item()
                total_samples += torso_labels.size(0)
            
            avg_loss = total_loss / len(dataloader)
            torso_acc = 100 * correct_torso / total_samples
            legs_acc = 100 * correct_legs / total_samples
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, '
                  f'Torso Acc: {torso_acc:.2f}%, Legs Acc: {legs_acc:.2f}%')
        
        # Guardar modelo
        torch.save(self.color_classifier.state_dict(), 'color_classifier.pth')
        print("Modelo de clasificación de colores guardado: color_classifier.pth")
    
    def load_color_classifier(self, model_path):
        """Carga el clasificador de colores entrenado"""
        self.color_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        self.color_classifier.eval()
        print(f"Clasificador de colores cargado desde: {model_path}")
    
    def detect_and_classify(self, image_path):
        """Detecta personas y clasifica colores de ropa"""
        # Cargar imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detección YOLO
        results = self.yolo_model(image_path)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filtrar solo personas (clase 0 en COCO)
                    if int(box.cls) == 0:  # Persona
                        # Extraer bbox
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Recortar región de la persona
                        person_crop = image_rgb[int(y1):int(y2), int(x1):int(x2)]
                        
                        if person_crop.size > 0:
                            # Clasificar colores
                            torso_color, legs_color = self.classify_colors(person_crop)
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': float(confidence),
                                'torso_color': torso_color,
                                'legs_color': legs_color
                            })
        
        return detections
    
    def classify_colors(self, person_crop):
        """Clasifica colores de torso y piernas"""
        self.color_classifier.eval()
        
        # Preprocesar imagen
        if isinstance(person_crop, np.ndarray):
            person_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            torso_pred, legs_pred = self.color_classifier(person_tensor)
            
            # Obtener predicciones
            torso_idx = torch.argmax(torso_pred, dim=1).cpu().numpy()[0]
            legs_idx = torch.argmax(legs_pred, dim=1).cpu().numpy()[0]
            
            torso_color = self.label_encoder.inverse_transform([torso_idx])[0]
            legs_color = self.label_encoder.inverse_transform([legs_idx])[0]
        
        return torso_color, legs_color
    
    def visualize_results(self, image_path, detections, save_path=None):
        """Visualiza los resultados de detección y clasificación"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Dibujar bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Añadir texto con colores
            text = f"Torso: {detection['torso_color']}\nPiernas: {detection['legs_color']}"
            plt.text(x, y-10, text, fontsize=10, color='red', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.axis('off')
        plt.title('Detección de Personas y Clasificación de Colores')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Resultado guardado en: {save_path}")
        
        plt.show()

def main():
    """Función principal para ejecutar el sistema"""
    # Crear sistema
    system = YOLOColorSystem()
    
    # Crear template de anotaciones si no existe
    if not os.path.exists('annotations_template.json'):
        create_annotations_template()
    
    # Entrenar clasificador de colores (descomenta si tienes datos)
    # system.train_color_classifier('path/to/your/images', 'annotations.json')
    
    # O cargar modelo preentrenado
    # system.load_color_classifier('color_classifier.pth')
    
    # Probar con imágenes
    test_images = ['test1.jpg', 'test2.jpg']  # Pon aquí tus imágenes de prueba
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nProcesando: {image_path}")
            detections = system.detect_and_classify(image_path)
            
            print(f"Encontradas {len(detections)} personas:")
            for i, detection in enumerate(detections):
                print(f"  Persona {i+1}: Torso={detection['torso_color']}, "
                      f"Piernas={detection['legs_color']}, Conf={detection['confidence']:.2f}")
            
            # Visualizar resultados
            system.visualize_results(image_path, detections, 
                                   save_path=f'result_{os.path.basename(image_path)}')
        else:
            print(f"Imagen no encontrada: {image_path}")

if __name__ == "__main__":
    main()

# Ejemplo de uso alternativo
"""
# Uso paso a paso:

# 1. Crear sistema
system = YOLOColorSystem()

# 2. Entrenar clasificador (si tienes datos anotados)
system.train_color_classifier('mi_dataset/imagenes', 'mi_dataset/annotations.json')

# 3. Cargar clasificador entrenado
system.load_color_classifier('color_classifier.pth')

# 4. Procesar imagen
detections = system.detect_and_classify('mi_foto.jpg')

# 5. Visualizar resultados
system.visualize_results('mi_foto.jpg', detections, 'resultado.jpg')
"""