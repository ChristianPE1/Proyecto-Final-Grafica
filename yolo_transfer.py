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
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from collections import Counter
import yaml
from pathlib import Path
import json
import pandas as pd
import random
from glob import glob

# Configuración de colores
COLORS = [
   'negro', 'blanco', 'gris', 'azul', 'azul_marino', 'azul_claro',
   'rojo', 'rosa', 'verde', 'verde_oscuro', 'amarillo', 'naranja',
   'marron', 'beige', 'violeta', 'morado', 'dorado', 'plateado'
]

# Configuración de categorías de ropa de DeepFashion2
DEEPFASHION2_CATEGORIES = {
    1: 'short_sleeved_shirt',
    2: 'long_sleeved_shirt', 
    3: 'short_sleeved_outwear',
    4: 'long_sleeved_outwear',
    5: 'vest',
    6: 'sling',
    7: 'shorts',
    8: 'trousers',
    9: 'skirt',
    10: 'short_sleeved_dress',
    11: 'long_sleeved_dress',
    12: 'vest_dress',
    13: 'sling_dress'
}

class SuperviselyDataset(Dataset):
   """Dataset para entrenar YOLO con el dataset Human Segmentation Dataset - Supervise.ly"""
   def __init__(self, images_dir, masks_dir, img_size=640):
      self.images_dir = Path(images_dir)
      self.masks_dir = Path(masks_dir)
      self.img_size = img_size

      # Obtener lista de imágenes válidas
      self.valid_samples = []
      for img_path in self.images_dir.glob('*.png'):
         # Buscar máscara correspondiente (puede ser .png o .jpg)
         mask_path_png = self.masks_dir / f"{img_path.stem}.png"
         mask_path_jpg = self.masks_dir / f"{img_path.stem}.jpg"
         
         if mask_path_png.exists():
               self.valid_samples.append((img_path, mask_path_png))
         elif mask_path_jpg.exists():
               self.valid_samples.append((img_path, mask_path_jpg))

      print(f"Encontradas {len(self.valid_samples)} imágenes válidas con máscaras")

   def __len__(self):
      return len(self.valid_samples)

   def __getitem__(self, idx):
      img_path, mask_path = self.valid_samples[idx]
      return str(img_path), str(mask_path)

class ColorExtractor:
   """Extractor de colores dominantes mejorado"""
   def __init__(self):
      self.color_names = COLORS

   def extract_dominant_color(self, image_region, n_colors=3):
      """Extrae el color dominante de una región"""
      if image_region.size == 0:
         return 'gris', [128, 128, 128]

      # Reshape para K-means
      pixels = image_region.reshape(-1, 3)

      # Remover píxeles muy oscuros o muy claros (posibles sombras/reflejos)
      pixels = pixels[np.all(pixels > 20, axis=1)]  # Remover muy oscuros
      pixels = pixels[np.all(pixels < 235, axis=1)]  # Remover muy claros

      if len(pixels) < 10:
         return 'gris', [128, 128, 128]

      # Aplicar K-means
      kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
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
      """Convierte RGB a nombre de color más preciso"""
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

class PersonSegmentationSystem:
   """Sistema completo: Segmentación de personas + extracción de colores"""

   def __init__(self, base_model='yolov8n-seg.pt'):
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      print(f"Usando dispositivo: {self.device}")

      # Modelo YOLO de segmentación
      self.yolo_model = YOLO(base_model)
      self.color_extractor = ColorExtractor()

      # Configuración para crear dataset YOLO
      self.yolo_config = {
         'train': '',
         'val': '',
         'nc': 1,  # Solo clase persona
         'names': {0: 'person'}
      }

   def prepare_supervisely_dataset(self, images_dir, masks_dir, output_dir):
      """Convierte dataset Supervise.ly para entrenar YOLO en segmentación de personas"""

      # Crear directorios de salida
      output_path = Path(output_dir)
      (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
      (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
      (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
      (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

      # Procesar imágenes
      images_dir = Path(images_dir)
      masks_dir = Path(masks_dir)

      processed_count = 0

      for img_path in images_dir.glob('*.png'):
         # Buscar máscara correspondiente
         mask_path_png = masks_dir / f"{img_path.stem}.png"
         mask_path_jpg = masks_dir / f"{img_path.stem}.jpg"
         
         mask_path = None
         if mask_path_png.exists():
               mask_path = mask_path_png
         elif mask_path_jpg.exists():
               mask_path = mask_path_jpg
         
         if mask_path is None:
               continue

         # Cargar imagen y máscara
         image = cv2.imread(str(img_path))
         mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
         
         if image is None or mask is None:
               continue

         h, w = image.shape[:2]

         # Binarizar máscara
         _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

         # Encontrar contornos de personas
         contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         if len(contours) == 0:
               continue

         # Determinar si va a train o val (80-20 split)
         subset = 'train' if processed_count % 5 != 0 else 'val'

         # Copiar imagen
         dst_img = output_path / 'images' / subset / img_path.name
         cv2.imwrite(str(dst_img), image)

         # Crear etiquetas YOLO para segmentación
         dst_label = output_path / 'labels' / subset / f"{img_path.stem}.txt"
         
         with open(dst_label, 'w') as f:
               for contour in contours:
                  # Obtener bbox del contorno
                  x, y, w_bbox, h_bbox = cv2.boundingRect(contour)
                  
                  # Filtrar contornos muy pequeños
                  if w_bbox < 20 or h_bbox < 20:
                     continue
                  
                  # Convertir contorno a coordenadas normalizadas
                  contour_normalized = []
                  for point in contour:
                     x_norm = point[0][0] / w
                     y_norm = point[0][1] / h
                     contour_normalized.extend([x_norm, y_norm])
                  
                  # Simplificar contorno si es muy complejo
                  if len(contour_normalized) > 200:  # Máximo 100 puntos
                     epsilon = 0.01 * cv2.arcLength(contour, True)
                     contour_simplified = cv2.approxPolyDP(contour, epsilon, True)
                     contour_normalized = []
                     for point in contour_simplified:
                           x_norm = point[0][0] / w
                           y_norm = point[0][1] / h
                           contour_normalized.extend([x_norm, y_norm])
                  
                  # Escribir en formato YOLO segmentación
                  if len(contour_normalized) >= 6:  # Mínimo 3 puntos
                     line = "0 " + " ".join(map(str, contour_normalized))
                     f.write(line + "\n")

         processed_count += 1

         if processed_count % 100 == 0:
               print(f"Procesadas {processed_count} imágenes...")

      # Crear archivo de configuración YOLO
      self.yolo_config['train'] = str(output_path / 'images' / 'train')
      self.yolo_config['val'] = str(output_path / 'images' / 'val')

      config_path = output_path / 'person_segmentation.yaml'
      with open(config_path, 'w') as f:
         yaml.dump(self.yolo_config, f)

      print(f"Dataset preparado: {processed_count} imágenes procesadas")
      print(f"Configuración guardada en: {config_path}")

      return str(config_path)

   def train_yolo_person_segmentation(self, dataset_config_path, epochs=100, batch_size=16):
      """Entrena YOLO para segmentación de personas"""

      print("Iniciando entrenamiento YOLO para segmentación de personas...")

      # Entrenar modelo
      results = self.yolo_model.train(
         data=dataset_config_path,
         epochs=epochs,
         batch=batch_size,
         imgsz=640,
         device=self.device,
         workers=4,
         patience=10,
         save=True,
         project='person_segmentation',
         name='yolo_person_seg_v1'
      )

      print("Entrenamiento completado!")

      # Mostrar loss y métricas por época
      results_csv = "person_segmentation/yolo_person_seg_v1/results.csv"
      if os.path.exists(results_csv):
         df = pd.read_csv(results_csv)
         print("\nResumen de entrenamiento (loss y métricas por época):")
         print(df[["epoch", "train/box_loss", "train/seg_loss", "metrics/mAP_0.5"]])
      else:
         print("No se encontró el archivo de resultados para mostrar métricas.")

      return results

   def load_trained_model(self, model_path):
      """Cargar modelo YOLO entrenado"""
      self.yolo_model = YOLO(model_path)
      print(f"Modelo cargado: {model_path}")

   def segment_and_extract_colors(self, image_path, conf_threshold=0.5):
      """Segmenta personas y extrae colores de torso y piernas usando las máscaras"""

      # Realizar segmentación
      results = self.yolo_model(image_path, conf=conf_threshold)

      # Cargar imagen original
      image = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      h, w = image_rgb.shape[:2]

      detections = []

      for result in results:
         if result.masks is not None:
               boxes = result.boxes
               masks = result.masks
               
               for i in range(len(boxes)):
                  # Obtener bbox y confianza
                  x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                  confidence = boxes.conf[i].cpu().numpy()
                  
                  # Obtener máscara
                  mask = masks.data[i].cpu().numpy()
                  
                  # Redimensionar máscara al tamaño original
                  mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                  
                  # Crear máscara binaria
                  mask_binary = (mask_resized > 0.5).astype(np.uint8)
                  
                  # Aplicar máscara a la imagen
                  person_pixels = image_rgb[mask_binary == 1]
                  
                  if len(person_pixels) > 0:
                     # Dividir la máscara en regiones de torso y piernas
                     mask_coords = np.where(mask_binary == 1)
                     y_coords = mask_coords[0]
                     
                     # Calcular límites para torso y piernas
                     y_min, y_max = np.min(y_coords), np.max(y_coords)
                     person_height = y_max - y_min
                     
                     # Torso: 60% superior
                     torso_limit = y_min + int(person_height * 0.6)
                     
                     # Crear máscaras para torso y piernas
                     torso_mask = mask_binary.copy()
                     torso_mask[torso_limit:, :] = 0
                     
                     legs_mask = mask_binary.copy()
                     legs_mask[:torso_limit, :] = 0
                     
                     # Extraer píxeles de torso y piernas
                     torso_pixels = image_rgb[torso_mask == 1]
                     legs_pixels = image_rgb[legs_mask == 1]
                     
                     # Extraer colores dominantes
                     torso_color, torso_rgb = self.color_extractor.extract_dominant_color(torso_pixels)
                     legs_color, legs_rgb = self.color_extractor.extract_dominant_color(legs_pixels)
                     
                     detections.append({
                           'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                           'confidence': float(confidence),
                           'mask': mask_binary,
                           'torso_color': torso_color,
                           'torso_rgb': torso_rgb.tolist(),
                           'legs_color': legs_color,
                           'legs_rgb': legs_rgb.tolist()
                     })

      return detections

   def visualize_segmentation_results(self, image_path, detections, save_path=None):
      """Visualiza resultados de segmentación y colores mejorado"""

      image = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      fig, axes = plt.subplots(1, 2, figsize=(20, 10))
      
      # Imagen original con bboxes
      axes[0].imshow(image_rgb)
      axes[0].set_title('Detección de Personas', fontsize=14)
      
      # Imagen con segmentación limpia
      axes[1].set_title('Segmentación y Colores Extraídos', fontsize=14)

      # Crear fondo oscurecido para la segunda imagen
      darkened_background = image_rgb * 0.3  # Oscurecer fondo al 30%
      result_image = darkened_background.copy()

      for i, detection in enumerate(detections):
         bbox = detection['bbox']
         x, y, w, h = bbox
         mask = detection['mask']

         # Dibujar bbox en imagen original
         rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=3)
         axes[0].add_patch(rect)

         # En la segunda imagen, mantener colores originales de la persona
         person_region = image_rgb * np.expand_dims(mask, axis=2)
         result_image[mask == 1] = person_region[mask == 1]

         # Información de colores (confidence es la probabilidad de detección)
         confidence = detection['confidence']
         torso_color = detection['torso_color']
         legs_color = detection['legs_color']

         text = f"Persona {i+1}\nTorso: {torso_color}\nPiernas: {legs_color}\nConfianza: {confidence:.2f}"
         
         axes[0].text(x, y-60, text, fontsize=10, color='white', weight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.8))

         # Agregar información de colores en la segunda imagen también
         axes[1].text(x, y-60, f"Persona {i+1}\nTorso: {torso_color}\nPiernas: {legs_color}", 
                     fontsize=10, color='white', weight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="blue", alpha=0.8))

      # Mostrar imagen con segmentación limpia
      axes[1].imshow(result_image.astype(np.uint8))
      
      for ax in axes:
         ax.axis('off')

      plt.tight_layout()

      if save_path:
         plt.savefig(save_path, bbox_inches='tight', dpi=300)
         print(f"Resultado guardado: {save_path}")

      plt.show()

   def batch_process_images(self, images_dir, output_dir):
      """Procesa múltiples imágenes y guarda resultados"""

      images_dir = Path(images_dir)
      output_dir = Path(output_dir)
      output_dir.mkdir(exist_ok=True)

      results_summary = []

      for img_path in images_dir.glob('*.png'):
         print(f"Procesando: {img_path.name}")

         # Segmentar y extraer colores
         detections = self.segment_and_extract_colors(str(img_path))

         # Guardar visualización
         save_path = output_dir / f"segmentation_result_{img_path.name}"
         self.visualize_segmentation_results(str(img_path), detections, str(save_path))

         # Preparar datos para JSON (sin la máscara binaria)
         json_detections = []
         for det in detections:
               json_det = det.copy()
               json_det.pop('mask', None)  # Remover máscara para JSON
               json_detections.append(json_det)

         # Guardar datos en JSON
         result_data = {
               'image': img_path.name,
               'detections': json_detections,
               'total_persons': len(detections)
         }

         results_summary.append(result_data)

         # Mostrar resumen
         print(f"  Detectadas {len(detections)} personas")
         for i, det in enumerate(detections):
               print(f"    Persona {i+1}: Torso={det['torso_color']}, "
                     f"Piernas={det['legs_color']}, Conf={det['confidence']:.2f}")

      # Guardar resumen completo
      with open(output_dir / 'segmentation_results_summary.json', 'w') as f:
         json.dump(results_summary, f, indent=2)

      print(f"\nProcesamiento completado. Resultados en: {output_dir}")
      return results_summary

class DeepFashion2Dataset(Dataset):
    """Dataset para entrenar YOLO con DeepFashion2 para detección de prendas"""
    def __init__(self, images_dir, annotations_dir, max_samples=5000, img_size=640):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.img_size = img_size
        self.max_samples = max_samples
        
        # Obtener lista de archivos de anotaciones
        annotation_files = list(self.annotations_dir.glob('*.json'))
        
        # Limitar número de muestras
        if len(annotation_files) > max_samples:
            annotation_files = random.sample(annotation_files, max_samples)
        
        self.valid_samples = []
        
        for ann_file in annotation_files:
            img_file = self.images_dir / f"{ann_file.stem}.jpg"
            if img_file.exists():
                self.valid_samples.append((str(img_file), str(ann_file)))
        
        print(f"DeepFashion2: Encontradas {len(self.valid_samples)} imágenes válidas para entrenamiento")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_path, ann_path = self.valid_samples[idx]
        return img_path, ann_path

class ClothingDetectionSystem:
    """Sistema para detectar prendas específicas usando DeepFashion2"""
    
    def __init__(self, base_model='yolov8n.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sistema de detección de prendas usando dispositivo: {self.device}")
        
        # Modelo YOLO para detección de prendas
        self.clothing_model = YOLO(base_model)
        
        # Configuración para DeepFashion2
        self.clothing_config = {
            'train': '',
            'val': '',
            'nc': 13,  # 13 categorías de ropa
            'names': DEEPFASHION2_CATEGORIES
        }
    
    def prepare_deepfashion2_dataset(self, train_images_dir, train_annotations_dir, 
                                   val_images_dir, val_annotations_dir, 
                                   output_dir, max_train=5000, max_val=500):
        """Convierte DeepFashion2 para entrenar YOLO en detección de prendas"""
        
        output_path = Path(output_dir)
        (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Procesar training set
        print("Procesando training set...")
        self._process_deepfashion2_subset(train_images_dir, train_annotations_dir,
                                        output_path / 'images' / 'train',
                                        output_path / 'labels' / 'train',
                                        max_samples=max_train)
        
        # Procesar validation set
        print("Procesando validation set...")
        self._process_deepfashion2_subset(val_images_dir, val_annotations_dir,
                                        output_path / 'images' / 'val',
                                        output_path / 'labels' / 'val',
                                        max_samples=max_val)
        
        # Crear archivo de configuración YOLO
        self.clothing_config['train'] = str(output_path / 'images' / 'train')
        self.clothing_config['val'] = str(output_path / 'images' / 'val')
        
        config_path = output_path / 'deepfashion2_clothing.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.clothing_config, f)
        
        print(f"Dataset DeepFashion2 preparado")
        print(f"Configuración guardada en: {config_path}")
        
        return str(config_path)
    
    def _process_deepfashion2_subset(self, images_dir, annotations_dir, 
                                   output_images_dir, output_labels_dir, max_samples):
        """Procesa un subset de DeepFashion2"""
        
        images_dir = Path(images_dir)
        annotations_dir = Path(annotations_dir)
        
        # Obtener archivos de anotaciones
        annotation_files = list(annotations_dir.glob('*.json'))
        
        if len(annotation_files) > max_samples:
            annotation_files = random.sample(annotation_files, max_samples)
        
        processed_count = 0
        
        for ann_file in annotation_files:
            img_file = images_dir / f"{ann_file.stem}.jpg"
            
            if not img_file.exists():
                continue
            
            # Cargar anotación
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            # Cargar imagen para obtener dimensiones
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Procesar cada item de ropa en la imagen
            yolo_labels = []
            
            for item in annotation.get('items', []):
                category_id = item.get('category_id', 0)
                
                # Solo procesar categorías válidas (1-13)
                if category_id < 1 or category_id > 13:
                    continue
                
                bbox = item.get('bounding_box', [])
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # Convertir a formato YOLO (centro, ancho, alto normalizados)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Clase en formato YOLO (0-indexed)
                class_id = category_id - 1
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Solo guardar si hay al menos una prenda válida
            if yolo_labels:
                # Copiar imagen
                dst_img = output_images_dir / img_file.name
                cv2.imwrite(str(dst_img), image)
                
                # Guardar etiquetas YOLO
                dst_label = output_labels_dir / f"{img_file.stem}.txt"
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_labels) + '\n')
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Procesadas {processed_count} imágenes...")
        
        print(f"Subset procesado: {processed_count} imágenes")
    
    def train_clothing_detection(self, dataset_config_path, epochs=50, batch_size=16):
        """Entrena YOLO para detección de prendas"""
        
        print("Iniciando entrenamiento YOLO para detección de prendas...")
        
        results = self.clothing_model.train(
            data=dataset_config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=self.device,
            workers=4,
            patience=10,
            save=True,
            project='clothing_detection',
            name='yolo_clothing_v1'
        )
        
        print("Entrenamiento de detección de prendas completado!")
        
        # Mostrar métricas
        results_csv = "clothing_detection/yolo_clothing_v1/results.csv"
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            print("\nResumen de entrenamiento (detección de prendas):")
            print(df[["epoch", "train/box_loss", "train/cls_loss", "metrics/mAP_0.5"]])
        
        return results
    
    def load_clothing_model(self, model_path):
        """Cargar modelo entrenado para detección de prendas"""
        self.clothing_model = YOLO(model_path)
        print(f"Modelo de detección de prendas cargado: {model_path}")
    
    def detect_clothing_items(self, person_crop, conf_threshold=0.5):
        """Detecta prendas en un recorte de persona"""
        
        results = self.clothing_model(person_crop, conf=conf_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Obtener nombre de la categoría
                    category_name = DEEPFASHION2_CATEGORIES.get(class_id + 1, 'unknown')
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'confidence': float(confidence),
                        'category_id': class_id,
                        'category_name': category_name
                    })
        
        return detections

# Ejemplo de uso completo
def main_person_segmentation():
   """Función para entrenar solo segmentación de personas"""

   # Configurar rutas para el dataset Human Segmentation Dataset - Supervise.ly
   SUPERVISELY_ROOT = "/content/human_segmentation_dataset"
   SUPERVISELY_IMAGES = f"{SUPERVISELY_ROOT}/images"
   SUPERVISELY_MASKS = f"{SUPERVISELY_ROOT}/masks"

   PERSON_DATASET_DIR = "/content/person_segmentation_dataset"

   # Crear sistema
   system = PersonSegmentationSystem()

   # Paso 1: Preparar dataset de segmentación desde Supervise.ly
   print("=== PASO 1: Preparando dataset de segmentación ===")
   dataset_config = system.prepare_supervisely_dataset(
      SUPERVISELY_IMAGES,
      SUPERVISELY_MASKS,
      PERSON_DATASET_DIR
   )

   # Paso 2: Entrenar YOLO para segmentación de personas
   print("\n=== PASO 2: Entrenando YOLO para segmentación ===")
   system.train_yolo_person_segmentation(
      dataset_config,
      epochs=10,
      batch_size=8
   )

   print("\n=== ENTRENAMIENTO DE PERSONAS COMPLETADO ===")

def main_clothing_detection():
   """Función para entrenar detección de prendas con DeepFashion2"""
   
   # Configurar rutas para DeepFashion2
   DEEPFASHION2_ROOT = "/content/deepfashion2"
   TRAIN_IMAGES = f"{DEEPFASHION2_ROOT}/train/image"
   TRAIN_ANNOTATIONS = f"{DEEPFASHION2_ROOT}/train/annos" 
   VAL_IMAGES = f"{DEEPFASHION2_ROOT}/validation/image"
   VAL_ANNOTATIONS = f"{DEEPFASHION2_ROOT}/validation/annos"
   
   CLOTHING_DATASET_DIR = "/content/clothing_dataset"
   
   # Crear sistema de detección de prendas
   clothing_system = ClothingDetectionSystem()
   
   # Paso 1: Preparar dataset de DeepFashion2
   print("=== PASO 1: Preparando dataset DeepFashion2 ===")
   dataset_config = clothing_system.prepare_deepfashion2_dataset(
       TRAIN_IMAGES, TRAIN_ANNOTATIONS,
       VAL_IMAGES, VAL_ANNOTATIONS,
       CLOTHING_DATASET_DIR,
       max_train=5000,  # Limitar a 5000 imágenes de entrenamiento
       max_val=500      # Limitar a 500 imágenes de validación
   )
   
   # Paso 2: Entrenar YOLO para detección de prendas
   print("\n=== PASO 2: Entrenando YOLO para detección de prendas ===")
   clothing_system.train_clothing_detection(
       dataset_config,
       epochs=30,
       batch_size=16
   )
   
   print("\n=== ENTRENAMIENTO DE PRENDAS COMPLETADO ===")

def main_complete_pipeline():
   """Función principal para el pipeline completo"""

   # Rutas de modelos entrenados
   PERSON_MODEL = "/content/person_segmentation/yolo_person_seg_v13/weights/best.pt"  # Ajustar según tu modelo
   CLOTHING_MODEL = "/content/clothing_detection/yolo_clothing_v1/weights/best.pt"   # Ajustar según tu modelo
   
   TEST_IMAGES_DIR = "/content/test_images"
   OUTPUT_DIR = "/content/complete_results"

   # Crear sistema integrado
   print("=== INICIANDO SISTEMA INTEGRADO ===")
   integrated_system = IntegratedPersonClothingSystem(
       person_model_path=PERSON_MODEL,
       clothing_model_path=CLOTHING_MODEL
   )

   # Procesar imágenes de test
   print("\n=== PROCESANDO IMÁGENES CON SISTEMA COMPLETO ===")
   results = integrated_system.batch_process_complete(TEST_IMAGES_DIR, OUTPUT_DIR)

   print("\n=== PIPELINE COMPLETO TERMINADO ===")
   print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
   # Descomenta la función que quieras ejecutar:
   
   # 1. Para entrenar solo segmentación de personas:
   # main_person_segmentation()
   
   # 2. Para entrenar solo detección de prendas:
   # main_clothing_detection()
   
   # 3. Para usar el pipeline completo (requiere modelos entrenados):
   main_complete_pipeline()

class IntegratedPersonClothingSystem:
    """Sistema integrado: Segmentación de personas + Detección de prendas + Extracción de colores"""
    
    def __init__(self, person_model_path=None, clothing_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sistema integrado usando dispositivo: {self.device}")
        
        # Sistema de segmentación de personas
        self.person_system = PersonSegmentationSystem()
        if person_model_path:
            self.person_system.load_trained_model(person_model_path)
        
        # Sistema de detección de prendas
        self.clothing_system = ClothingDetectionSystem()
        if clothing_model_path:
            self.clothing_system.load_clothing_model(clothing_model_path)
        
        self.color_extractor = ColorExtractor()
    
    def process_image_complete(self, image_path, person_conf=0.5, clothing_conf=0.5):
        """Proceso completo: detecta personas, luego prendas, y extrae colores"""
        
        # Paso 1: Detectar personas
        person_detections = self.person_system.segment_and_extract_colors(image_path, person_conf)
        
        # Cargar imagen original
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        complete_results = []
        
        for i, person in enumerate(person_detections):
            person_bbox = person['bbox']
            x, y, w, h = person_bbox
            
            # Paso 2: Recortar región de la persona
            person_crop = image_rgb[y:y+h, x:x+w]
            
            # Paso 3: Detectar prendas en el recorte de la persona
            clothing_detections = self.clothing_system.detect_clothing_items(person_crop, clothing_conf)
            
            # Paso 4: Ajustar coordenadas de prendas al sistema de coordenadas original
            adjusted_clothing = []
            for clothing in clothing_detections:
                clothing_bbox = clothing['bbox']
                cx, cy, cw, ch = clothing_bbox
                
                # Ajustar coordenadas
                adjusted_bbox = [x + cx, y + cy, cw, ch]
                
                # Extraer región de la prenda para análisis de color
                garment_x1, garment_y1 = x + cx, y + cy
                garment_x2, garment_y2 = garment_x1 + cw, garment_y1 + ch
                
                # Asegurar que las coordenadas estén dentro de la imagen
                garment_x1 = max(0, garment_x1)
                garment_y1 = max(0, garment_y1)
                garment_x2 = min(image_rgb.shape[1], garment_x2)
                garment_y2 = min(image_rgb.shape[0], garment_y2)
                
                garment_region = image_rgb[garment_y1:garment_y2, garment_x1:garment_x2]
                
                # Extraer color dominante de la prenda
                garment_color, garment_rgb = self.color_extractor.extract_dominant_color(garment_region)
                
                adjusted_clothing.append({
                    'bbox': adjusted_bbox,
                    'confidence': clothing['confidence'],
                    'category_name': clothing['category_name'],
                    'category_id': clothing['category_id'],
                    'color': garment_color,
                    'color_rgb': garment_rgb.tolist()
                })
            
            complete_results.append({
                'person_id': i + 1,
                'person_bbox': person_bbox,
                'person_confidence': person['confidence'],
                'person_mask': person['mask'],
                'torso_color': person['torso_color'],
                'legs_color': person['legs_color'],
                'clothing_items': adjusted_clothing
            })
        
        return complete_results
    
    def visualize_complete_results(self, image_path, results, save_path=None):
        """Visualiza resultados completos con personas y prendas detectadas"""
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        # Imagen original
        axes[0].imshow(image_rgb)
        axes[0].set_title('Imagen Original', fontsize=14)
        
        # Detección de personas
        axes[1].imshow(image_rgb)
        axes[1].set_title('Segmentación de Personas', fontsize=14)
        
        # Detección completa (personas + prendas)
        axes[2].imshow(image_rgb)
        axes[2].set_title('Detección Completa: Personas + Prendas', fontsize=14)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        for i, result in enumerate(results):
            person_bbox = result['person_bbox']
            px, py, pw, ph = person_bbox
            person_mask = result['person_mask']
            
            # Color para esta persona
            person_color = colors[i % len(colors)]
            
            # Dibujar persona en imagen de segmentación
            person_rect = plt.Rectangle((px, py), pw, ph, fill=False, 
                                      color=person_color, linewidth=3)
            axes[1].add_patch(person_rect)
            
            # Información de persona
            person_text = f"Persona {result['person_id']}\nTorso: {result['torso_color']}\nPiernas: {result['legs_color']}"
            axes[1].text(px, py-80, person_text, fontsize=10, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=person_color, alpha=0.8))
            
            # Dibujar persona en imagen completa
            person_rect2 = plt.Rectangle((px, py), pw, ph, fill=False, 
                                       color=person_color, linewidth=2)
            axes[2].add_patch(person_rect2)
            
            # Dibujar prendas detectadas
            for j, clothing in enumerate(result['clothing_items']):
                clothing_bbox = clothing['bbox']
                cx, cy, cw, ch = clothing_bbox
                
                # Rectángulo para la prenda
                clothing_rect = plt.Rectangle((cx, cy), cw, ch, fill=False, 
                                            color='cyan', linewidth=2, linestyle='--')
                axes[2].add_patch(clothing_rect)
                
                # Información de la prenda
                clothing_text = f"{clothing['category_name']}\n{clothing['color']}\nConf: {clothing['confidence']:.2f}"
                axes[2].text(cx, cy-40, clothing_text, fontsize=8, color='white', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.8))
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Resultado completo guardado: {save_path}")
        
        plt.show()
    
    def batch_process_complete(self, images_dir, output_dir):
        """Procesa múltiples imágenes con el sistema completo"""
        
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results_summary = []
        
        for img_path in images_dir.glob('*.jpg'):
            print(f"Procesando imagen completa: {img_path.name}")
            
            # Proceso completo
            results = self.process_image_complete(str(img_path))
            
            # Visualizar
            save_path = output_dir / f"complete_result_{img_path.name}"
            self.visualize_complete_results(str(img_path), results, str(save_path))
            
            # Preparar datos para JSON
            json_results = []
            for result in results:
                json_result = result.copy()
                json_result.pop('person_mask', None)  # Remover máscara para JSON
                json_results.append(json_result)
            
            result_data = {
                'image': img_path.name,
                'total_persons': len(results),
                'detections': json_results
            }
            
            results_summary.append(result_data)
            
            # Mostrar resumen
            print(f"  Detectadas {len(results)} personas")
            for result in results:
                print(f"    Persona {result['person_id']}: {len(result['clothing_items'])} prendas detectadas")
                for clothing in result['clothing_items']:
                    print(f"      - {clothing['category_name']} ({clothing['color']})")
        
        # Guardar resumen completo
        with open(output_dir / 'complete_results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nProcesamiento completo terminado. Resultados en: {output_dir}")
        return results_summary