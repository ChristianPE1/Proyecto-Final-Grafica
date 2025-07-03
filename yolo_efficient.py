

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
import yaml
from pathlib import Path
import webcolors
from sklearn.cluster import KMeans
from collections import Counter

# Configuración de colores más completa
COLORS = [
    'negro', 'blanco', 'gris', 'azul', 'azul_marino', 'azul_claro',
    'rojo', 'rosa', 'verde', 'verde_oscuro', 'amarillo', 'naranja',
    'marron', 'beige', 'violeta', 'morado', 'dorado', 'plateado'
]

class FashionpediaParser:
    """Parser para el dataset Fashionpedia"""
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Mapear categorías de ropa a regiones corporales
        self.torso_categories = {
            'shirt', 'blouse', 'top', 'tank_top', 't-shirt', 'sweater', 
            'hoodie', 'jacket', 'blazer', 'vest', 'dress', 'jumpsuit'
        }
        
        self.legs_categories = {
            'pants', 'jeans', 'trousers', 'shorts', 'skirt', 'leggings',
            'dress', 'jumpsuit'  # dress/jumpsuit pueden ser ambos
        }
        
        # Crear mapeo inverso de IDs a nombres
        self.id_to_name = {v: k for k, v in self.config.get('names', {}).items()}
    
    def parse_label_line(self, line):
        """Parse una línea del archivo de etiquetas YOLO"""
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        return {
            'class_id': class_id,
            'class_name': self.id_to_name.get(class_id, 'unknown'),
            'bbox': [x_center, y_center, width, height],  # YOLO format
            'region': self.get_clothing_region(class_id)
        }
    
    def get_clothing_region(self, class_id):
        """Determina si la prenda es para torso, piernas o ambos"""
        class_name = self.id_to_name.get(class_id, '').lower()
        
        if any(cat in class_name for cat in self.torso_categories):
            if class_name in ['dress', 'jumpsuit']:
                return 'both'
            return 'torso'
        elif any(cat in class_name for cat in self.legs_categories):
            if class_name in ['dress', 'jumpsuit']:
                return 'both'
            return 'legs'
        else:
            return 'unknown'

class ColorExtractor:
    """Extractor de colores dominantes de prendas"""
    def __init__(self):
        self.color_names = COLORS
        
    def extract_dominant_color(self, image_region, n_colors=3):
        """Extrae el color dominante de una región de imagen"""
        # Reshape para K-means
        pixels = image_region.reshape(-1, 3)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Obtener colores y sus frecuencias
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Contar frecuencias
        label_counts = Counter(labels)
        
        # Obtener el color más frecuente
        dominant_idx = label_counts.most_common(1)[0][0]
        dominant_color_rgb = colors[dominant_idx].astype(int)
        
        # Convertir RGB a nombre de color
        color_name = self.rgb_to_color_name(dominant_color_rgb)
        
        return color_name, dominant_color_rgb
    
    def rgb_to_color_name(self, rgb):
        """Convierte RGB a nombre de color aproximado"""
        try:
            # Intentar obtener nombre exacto
            color_name = webcolors.rgb_to_name(rgb)
            return self.map_to_our_colors(color_name)
        except ValueError:
            # Si no hay nombre exacto, usar distancia mínima
            return self.closest_color(rgb)
    
    def closest_color(self, rgb):
        """Encuentra el color más cercano en nuestra paleta"""
        color_map = {
            'negro': [0, 0, 0],
            'blanco': [255, 255, 255],
            'gris': [128, 128, 128],
            'azul': [0, 0, 255],
            'azul_marino': [0, 0, 128],
            'azul_claro': [173, 216, 230],
            'rojo': [255, 0, 0],
            'rosa': [255, 192, 203],
            'verde': [0, 255, 0],
            'verde_oscuro': [0, 100, 0],
            'amarillo': [255, 255, 0],
            'naranja': [255, 165, 0],
            'marron': [165, 42, 42],
            'beige': [245, 245, 220],
            'violeta': [238, 130, 238],
            'morado': [128, 0, 128],
            'dorado': [255, 215, 0],
            'plateado': [192, 192, 192]
        }
        
        min_distance = float('inf')
        closest_color_name = 'gris'
        
        for color_name, color_rgb in color_map.items():
            distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, color_rgb)))
            if distance < min_distance:
                min_distance = distance
                closest_color_name = color_name
        
        return closest_color_name
    
    def map_to_our_colors(self, webcolor_name):
        """Mapea nombres de webcolors a nuestros colores"""
        color_mapping = {
            'black': 'negro', 'white': 'blanco', 'gray': 'gris', 'grey': 'gris',
            'blue': 'azul', 'navy': 'azul_marino', 'lightblue': 'azul_claro',
            'red': 'rojo', 'pink': 'rosa', 'green': 'verde', 'darkgreen': 'verde_oscuro',
            'yellow': 'amarillo', 'orange': 'naranja', 'brown': 'marron',
            'beige': 'beige', 'violet': 'violeta', 'purple': 'morado',
            'gold': 'dorado', 'silver': 'plateado'
        }
        return color_mapping.get(webcolor_name.lower(), 'gris')

class FashionpediaDataset(Dataset):
    """Dataset para el dataset Fashionpedia con extracción de colores"""
    def __init__(self, images_dir, labels_dir, yaml_path, transform=None, extract_colors=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.extract_colors = extract_colors
        
        # Parser de Fashionpedia
        self.parser = FashionpediaParser(yaml_path)
        self.color_extractor = ColorExtractor()
        
        # Obtener lista de imágenes
        self.image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        
        # Filtrar imágenes que tienen etiquetas correspondientes
        self.valid_samples = []
        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.valid_samples.append((img_path, label_path))
        
        print(f"Encontradas {len(self.valid_samples)} imágenes válidas con etiquetas")
        
        # Encoder para colores
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(COLORS)
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.valid_samples[idx]
        
        # Cargar imagen
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Leer etiquetas
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Procesar cada prenda
        torso_colors = []
        legs_colors = []
        
        for line in lines:
            item = self.parser.parse_label_line(line)
            if item is None:
                continue
            
            # Convertir bbox de YOLO a píxeles
            x_center, y_center, width, height = item['bbox']
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # Asegurar que las coordenadas estén dentro de la imagen
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extraer región de la prenda
            garment_region = image[y1:y2, x1:x2]
            
            if garment_region.size > 0 and self.extract_colors:
                # Extraer color dominante
                color_name, _ = self.color_extractor.extract_dominant_color(garment_region)
                
                # Asignar a torso o piernas según el tipo de prenda
                if item['region'] == 'torso':
                    torso_colors.append(color_name)
                elif item['region'] == 'legs':
                    legs_colors.append(color_name)
                elif item['region'] == 'both':
                    torso_colors.append(color_name)
                    legs_colors.append(color_name)
        
        # Obtener color más frecuente para cada región
        torso_color = Counter(torso_colors).most_common(1)[0][0] if torso_colors else 'gris'
        legs_color = Counter(legs_colors).most_common(1)[0][0] if legs_colors else 'gris'
        
        # Transformar imagen si es necesario
        if self.transform:
            image = self.transform(image)
        
        # Codificar colores
        torso_encoded = self.label_encoder.transform([torso_color])[0]
        legs_encoded = self.label_encoder.transform([legs_color])[0]
        
        return image, torso_encoded, legs_encoded, str(img_path)

class ImprovedColorClassifier(nn.Module):
    """Clasificador mejorado con atención espacial"""
    def __init__(self, num_classes=len(COLORS)):
        super(ImprovedColorClassifier, self).__init__()
        
        # Backbone pre-entrenado más pequeño para acelerar
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Obtener dimensiones de características
        num_features = self.backbone.classifier[-1].in_features
        
        # Reemplazar clasificador
        self.backbone.classifier = nn.Identity()
        
        # Módulo de atención espacial
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Clasificadores separados
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.torso_classifier = nn.Linear(512, num_classes)
        self.legs_classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Extraer características del backbone
        features = self.backbone.features(x)
        
        # Aplicar atención espacial
        attention_weights = self.spatial_attention(features)
        attended_features = features * attention_weights
        
        # Extraer características finales
        final_features = self.feature_extractor(attended_features)
        
        # Clasificar
        torso_logits = self.torso_classifier(final_features)
        legs_logits = self.legs_classifier(final_features)
        
        return torso_logits, legs_logits

class FashionpediaColorSystem:
    """Sistema completo usando Fashionpedia"""
    def __init__(self, yaml_path, yolo_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Cargar configuración de Fashionpedia
        self.parser = FashionpediaParser(yaml_path)
        self.color_extractor = ColorExtractor()
        
        # Modelo YOLO para detección de personas
        if yolo_model_path:
            self.yolo_model = YOLO(yolo_model_path)
        else:
            self.yolo_model = YOLO('yolov8n.pt')
        
        # Clasificador de colores mejorado
        self.color_classifier = ImprovedColorClassifier()
        self.color_classifier.to(self.device)
        
        # Transformaciones
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(COLORS)
    
    def train_on_fashionpedia(self, train_images_dir, train_labels_dir, 
                             val_images_dir, val_labels_dir, yaml_path,
                             epochs=50, batch_size=16, lr=0.001):
        """Entrena el modelo usando el dataset Fashionpedia"""
        print("Creando datasets...")
        
        # Crear datasets
        train_dataset = FashionpediaDataset(
            train_images_dir, train_labels_dir, yaml_path,
            transform=self.train_transform
        )
        
        val_dataset = FashionpediaDataset(
            val_images_dir, val_labels_dir, yaml_path,
            transform=self.val_transform
        )
        
        # Data loaders
        # Limitar a 5000 imágenes para entrenamiento rápido
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(5000, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(500, len(val_dataset))))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=2)
        
        # Optimizer y scheduler
        optimizer = optim.AdamW(self.color_classifier.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Variables para guardar el mejor modelo
        best_val_acc = 0.0
        
        print(f"Iniciando entrenamiento con {len(train_dataset)} muestras de entrenamiento...")
        
        for epoch in range(epochs):
            # Entrenamiento
            self.color_classifier.train()
            train_loss = 0.0
            train_correct_torso = 0
            train_correct_legs = 0
            train_total = 0
            
            for batch_idx, (images, torso_labels, legs_labels, _) in enumerate(train_loader):
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
                
                train_loss += loss.item()
                
                # Calcular precisión
                _, torso_predicted = torch.max(torso_pred, 1)
                _, legs_predicted = torch.max(legs_pred, 1)
                
                train_correct_torso += (torso_predicted == torso_labels).sum().item()
                train_correct_legs += (legs_predicted == legs_labels).sum().item()
                train_total += torso_labels.size(0)
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validación
            self.color_classifier.eval()
            val_loss = 0.0
            val_correct_torso = 0
            val_correct_legs = 0
            val_total = 0
            
            with torch.no_grad():
                for images, torso_labels, legs_labels, _ in val_loader:
                    images = images.to(self.device)
                    torso_labels = torso_labels.to(self.device)
                    legs_labels = legs_labels.to(self.device)
                    
                    torso_pred, legs_pred = self.color_classifier(images)
                    
                    loss_torso = criterion(torso_pred, torso_labels)
                    loss_legs = criterion(legs_pred, legs_labels)
                    loss = loss_torso + loss_legs
                    
                    val_loss += loss.item()
                    
                    _, torso_predicted = torch.max(torso_pred, 1)
                    _, legs_predicted = torch.max(legs_pred, 1)
                    
                    val_correct_torso += (torso_predicted == torso_labels).sum().item()
                    val_correct_legs += (legs_predicted == legs_labels).sum().item()
                    val_total += torso_labels.size(0)
            
            # Calcular métricas
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            
            train_acc_torso = 100 * train_correct_torso / train_total
            train_acc_legs = 100 * train_correct_legs / train_total
            val_acc_torso = 100 * val_correct_torso / val_total
            val_acc_legs = 100 * val_correct_legs / val_total
            
            val_acc_avg = (val_acc_torso + val_acc_legs) / 2
            
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}')
            print(f'  Train Acc - Torso: {train_acc_torso:.2f}%, Piernas: {train_acc_legs:.2f}%')
            print(f'  Val Acc - Torso: {val_acc_torso:.2f}%, Piernas: {val_acc_legs:.2f}%')
            
            # Guardar mejor modelo
            if val_acc_avg > best_val_acc:
                best_val_acc = val_acc_avg
                torch.save(self.color_classifier.state_dict(), 'best_fashionpedia_color_classifier.pth')
                print(f'  Nuevo mejor modelo guardado! Val Acc: {best_val_acc:.2f}%')
            
            scheduler.step()
            print()
        
        print(f"Entrenamiento completado. Mejor precisión de validación: {best_val_acc:.2f}%")
    
    def load_model(self, model_path):
        """Cargar modelo entrenado"""
        self.color_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        self.color_classifier.eval()
        print(f"Modelo cargado desde: {model_path}")
    
    def predict_image(self, image_path):
        """Predecir colores en una imagen"""
        # Detectar personas con YOLO
        results = self.yolo_model(image_path)
        
        # Cargar imagen original
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filtrar solo personas (clase 0)
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Recortar persona
                        person_crop = image_rgb[int(y1):int(y2), int(x1):int(x2)]
                        
                        if person_crop.size > 0:
                            # Clasificar colores
                            torso_color, legs_color = self.classify_person_colors(person_crop)
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': float(confidence),
                                'torso_color': torso_color,
                                'legs_color': legs_color
                            })
        
        return detections
    
    def classify_person_colors(self, person_crop):
        """Clasificar colores de una persona"""
        self.color_classifier.eval()
        
        # Preprocesar
        person_tensor = self.val_transform(person_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            torso_pred, legs_pred = self.color_classifier(person_tensor)
            
            torso_idx = torch.argmax(torso_pred, dim=1).cpu().numpy()[0]
            legs_idx = torch.argmax(legs_pred, dim=1).cpu().numpy()[0]
            
            torso_color = self.label_encoder.inverse_transform([torso_idx])[0]
            legs_color = self.label_encoder.inverse_transform([legs_idx])[0]
        
        return torso_color, legs_color
    
    def visualize_results(self, image_path, detections, save_path=None):
        """Visualizar resultados"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Dibujar bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Texto con colores
            text = f"Torso: {detection['torso_color']}\nPiernas: {detection['legs_color']}\nConf: {detection['confidence']:.2f}"
            plt.text(x, y-30, text, fontsize=9, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.axis('off')
        plt.title('Detección y Clasificación de Colores con Fashionpedia')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Resultado guardado: {save_path}")
        
        plt.show()

def main():
    """Función principal"""
    # Configurar rutas del dataset Fashionpedia
    DATASET_ROOT = "/content/fashionpedia"  # Cambiar por tu ruta
    yaml_path = f"{DATASET_ROOT}/data.yaml"
    
    train_images = f"{DATASET_ROOT}/images/train"
    train_labels = f"{DATASET_ROOT}/labels/train"
    val_images = f"{DATASET_ROOT}/images/val"
    val_labels = f"{DATASET_ROOT}/labels/val"
    
    # Crear sistema
    system = FashionpediaColorSystem(yaml_path)
    
    # Entrenar modelo
    print("Iniciando entrenamiento con Fashionpedia...")
    system.train_on_fashionpedia(
        train_images, train_labels,
        val_images, val_labels,
        yaml_path,
        epochs=5,         # Menos épocas para pruebas rápidas
        batch_size=4,     # Batch pequeño para menos RAM
        lr=0.0001
    )
    
    # Cargar mejor modelo
    system.load_model('best_fashionpedia_color_classifier.pth')
    
    # Probar con imágenes
    #test_images = ['test1.jpg', 'test2.jpg']
    import glob
    test_images = sorted(glob.glob('/content/test/*.jpg')) + sorted(glob.glob('/content/test/*.png'))
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nProcesando: {img_path}")
            detections = system.predict_image(img_path)
            
            print(f"Detectadas {len(detections)} personas:")
            for i, det in enumerate(detections):
                print(f"  Persona {i+1}: Torso={det['torso_color']}, "
                      f"Piernas={det['legs_color']}, Conf={det['confidence']:.2f}")
            
            system.visualize_results(img_path, detections, f'result_{os.path.basename(img_path)}')

if __name__ == "__main__":
    main()

# Ejemplo de uso paso a paso
"""
# 1. Descargar Fashionpedia dataset de Kaggle
# 2. Extraer en una carpeta con estructura:
#    fashionpedia/
#    ├── images/
#    │   ├── train/
#    │   ├── val/
#    │   └── test/
#    ├── labels/
#    │   ├── train/
#    │   ├── val/
#    │   └── test/
#    └── fashionpedia.yaml

# 3. Usar el sistema:
system = FashionpediaColorSystem('fashionpedia/fashionpedia.yaml')

# 4. Entrenar:
system.train_on_fashionpedia(
    'fashionpedia/images/train', 'fashionpedia/labels/train',
    'fashionpedia/images/val', 'fashionpedia/labels/val',
    'fashionpedia/fashionpedia.yaml'
)

# 5. Cargar y probar:
system.load_model('best_fashionpedia_color_classifier.pth')
detections = system.predict_image('mi_foto.jpg')
system.visualize_results('mi_foto.jpg', detections)
"""