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

# Configuración de colores
COLORS = [
   'negro', 'blanco', 'gris', 'azul', 'azul_marino', 'azul_claro',
   'rojo', 'rosa', 'verde', 'verde_oscuro', 'amarillo', 'naranja',
   'marron', 'beige', 'violeta', 'morado', 'dorado', 'plateado'
]

class PersonDataset(Dataset):
   """Dataset para entrenar YOLO en detección de personas"""
   def __init__(self, images_dir, labels_dir, img_size=640):
      self.images_dir = Path(images_dir)
      self.labels_dir = Path(labels_dir)
      self.img_size = img_size

      # Obtener lista de imágenes válidas
      self.valid_samples = []
      for img_path in self.images_dir.glob('*.jpg'):
         label_path = self.labels_dir / f"{img_path.stem}.txt"
         if label_path.exists():
               self.valid_samples.append((img_path, label_path))

      print(f"Encontradas {len(self.valid_samples)} imágenes válidas para personas")

   def __len__(self):
      return len(self.valid_samples)

   def __getitem__(self, idx):
      img_path, label_path = self.valid_samples[idx]
      return str(img_path), str(label_path)

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

class PersonColorDetectionSystem:
   """Sistema completo: YOLO para personas + extracción de colores"""

   def __init__(self, base_model='yolov8n.pt'):
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      print(f"Usando dispositivo: {self.device}")

      # Modelo YOLO base
      self.yolo_model = YOLO(base_model)
      self.color_extractor = ColorExtractor()

      # Configuración para crear dataset YOLO
      self.yolo_config = {
         'train': '',
         'val': '',
         'nc': 1,  # Solo clase persona
         'names': {0: 'person'}
      }

   def prepare_person_dataset(self, fashionpedia_images_dir, fashionpedia_labels_dir,
                           yaml_path, output_dir):
      """Convierte dataset Fashionpedia para entrenar YOLO en detección de personas"""

      # Crear directorios de salida
      output_path = Path(output_dir)
      (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
      (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
      (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
      (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

      # Cargar configuración original
      with open(yaml_path, 'r') as f:
         original_config = yaml.safe_load(f)

      # Procesar imágenes
      images_dir = Path(fashionpedia_images_dir)
      labels_dir = Path(fashionpedia_labels_dir)

      processed_count = 0

      for img_path in images_dir.glob('*.jpg'):
         label_path = labels_dir / f"{img_path.stem}.txt"

         if not label_path.exists():
               continue

         # Leer etiquetas originales
         with open(label_path, 'r') as f:
               lines = f.readlines()

         # Buscar si hay al menos una prenda (indicando presencia de persona)
         has_clothing = False
         for line in lines:
               parts = line.strip().split()
               if len(parts) >= 5:
                  has_clothing = True
                  break

         if not has_clothing:
               continue

         # Cargar imagen para obtener dimensiones
         image = cv2.imread(str(img_path))
         if image is None:
               continue

         h, w = image.shape[:2]

         # Estimar bbox de la persona basándose en las prendas
         all_boxes = []
         for line in lines:
               parts = line.strip().split()
               if len(parts) >= 5:
                  x_center = float(parts[1])
                  y_center = float(parts[2])
                  width = float(parts[3])
                  height = float(parts[4])
                  all_boxes.append([x_center, y_center, width, height])

         if not all_boxes:
               continue

         # Calcular bbox que englobe todas las prendas
         all_boxes = np.array(all_boxes)
         x_centers = all_boxes[:, 0]
         y_centers = all_boxes[:, 1]
         widths = all_boxes[:, 2]
         heights = all_boxes[:, 3]

         # Coordenadas extremas
         x_mins = x_centers - widths/2
         x_maxs = x_centers + widths/2
         y_mins = y_centers - heights/2
         y_maxs = y_centers + heights/2

         # Bbox de la persona
         person_x_min = np.min(x_mins)
         person_x_max = np.max(x_maxs)
         person_y_min = np.min(y_mins)
         person_y_max = np.max(y_maxs)

         # Expandir bbox para incluir toda la persona
         person_width = person_x_max - person_x_min
         person_height = person_y_max - person_y_min

         # Expandir un poco más
         expansion = 0.1
         person_x_min = max(0, person_x_min - person_width * expansion)
         person_x_max = min(1, person_x_max + person_width * expansion)
         person_y_min = max(0, person_y_min - person_height * expansion)
         person_y_max = min(1, person_y_max + person_height * expansion)

         # Calcular centro y dimensiones finales
         person_x_center = (person_x_min + person_x_max) / 2
         person_y_center = (person_y_min + person_y_max) / 2
         person_width = person_x_max - person_x_min
         person_height = person_y_max - person_y_min

         # Determinar si va a train o val (80-20 split)
         subset = 'train' if processed_count % 5 != 0 else 'val'

         # Copiar imagen
         dst_img = output_path / 'images' / subset / img_path.name
         cv2.imwrite(str(dst_img), image)

         # Crear etiqueta para persona
         dst_label = output_path / 'labels' / subset / f"{img_path.stem}.txt"
         with open(dst_label, 'w') as f:
               f.write(f"0 {person_x_center:.6f} {person_y_center:.6f} {person_width:.6f} {person_height:.6f}\n")

         processed_count += 1

         if processed_count % 100 == 0:
               print(f"Procesadas {processed_count} imágenes...")

      # Crear archivo de configuración YOLO
      self.yolo_config['train'] = str(output_path / 'images' / 'train')
      self.yolo_config['val'] = str(output_path / 'images' / 'val')

      config_path = output_path / 'person_dataset.yaml'
      with open(config_path, 'w') as f:
         yaml.dump(self.yolo_config, f)

      print(f"Dataset preparado: {processed_count} imágenes procesadas")
      print(f"Configuración guardada en: {config_path}")

      return str(config_path)

   def train_yolo_person_detection(self, dataset_config_path, epochs=100, batch_size=16):
      """Entrena YOLO para detección de personas usando transfer learning"""

      print("Iniciando entrenamiento YOLO para detección de personas...")

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
         project='person_detection',
         name='yolo_person_v1'
      )

      print("Entrenamiento completado!")
      return results

   def load_trained_model(self, model_path):
      """Cargar modelo YOLO entrenado"""
      self.yolo_model = YOLO(model_path)
      print(f"Modelo cargado: {model_path}")

   def detect_and_extract_colors(self, image_path, conf_threshold=0.5):
      """Detecta personas y extrae colores de torso y piernas"""

      # Detectar personas
      results = self.yolo_model(image_path, conf=conf_threshold)

      # Cargar imagen original
      image = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      h, w = image_rgb.shape[:2]

      detections = []

      for result in results:
         boxes = result.boxes
         if boxes is not None:
               for box in boxes:
                  # Obtener coordenadas
                  x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                  confidence = box.conf[0].cpu().numpy()

                  # Asegurar que las coordenadas estén dentro de la imagen
                  x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))

                  # Extraer región de la persona
                  person_region = image_rgb[y1:y2, x1:x2]

                  if person_region.size > 0:
                     # Dividir en torso y piernas
                     person_h = y2 - y1

                     # Torso: 30% superior de la persona
                     torso_region = person_region[:int(person_h * 0.6)]

                     # Piernas: 40% inferior de la persona
                     legs_region = person_region[int(person_h * 0.6):]

                     # Extraer colores
                     torso_color, torso_rgb = self.color_extractor.extract_dominant_color(torso_region)
                     legs_color, legs_rgb = self.color_extractor.extract_dominant_color(legs_region)

                     detections.append({
                           'bbox': [x1, y1, x2-x1, y2-y1],
                           'confidence': float(confidence),
                           'torso_color': torso_color,
                           'torso_rgb': torso_rgb.tolist(),
                           'legs_color': legs_color,
                           'legs_rgb': legs_rgb.tolist()
                     })

      return detections

   def visualize_results(self, image_path, detections, save_path=None):
      """Visualiza resultados de detección y colores"""

      image = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      plt.figure(figsize=(15, 10))
      plt.imshow(image_rgb)

      for i, detection in enumerate(detections):
         bbox = detection['bbox']
         x, y, w, h = bbox

         # Dibujar bbox
         rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=3)
         plt.gca().add_patch(rect)

         # Mostrar colores extraídos
         torso_color = detection['torso_color']
         legs_color = detection['legs_color']
         confidence = detection['confidence']

         # Texto con información
         text = f"Persona {i+1}\nTorso: {torso_color}\nPiernas: {legs_color}\nConf: {confidence:.2f}"

         plt.text(x, y-40, text, fontsize=10, color='white', weight='bold',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.8))

         # Mostrar muestras de color
         torso_rgb = np.array(detection['torso_rgb']) / 255.0
         legs_rgb = np.array(detection['legs_rgb']) / 255.0

         # Cuadrados de color
         torso_square = plt.Rectangle((x + w - 60, y + 10), 25, 25,
                                    facecolor=torso_rgb, edgecolor='white', linewidth=2)
         legs_square = plt.Rectangle((x + w - 30, y + 10), 25, 25,
                                    facecolor=legs_rgb, edgecolor='white', linewidth=2)

         plt.gca().add_patch(torso_square)
         plt.gca().add_patch(legs_square)

      plt.axis('off')
      plt.title('Detección de Personas y Extracción de Colores de Ropa', fontsize=16)
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

      for img_path in images_dir.glob('*.jpg'):
         print(f"Procesando: {img_path.name}")

         # Detectar y extraer colores
         detections = self.detect_and_extract_colors(str(img_path))

         # Guardar visualización
         save_path = output_dir / f"result_{img_path.name}"
         self.visualize_results(str(img_path), detections, str(save_path))

         # Guardar datos en JSON
         result_data = {
               'image': img_path.name,
               'detections': detections,
               'total_persons': len(detections)
         }

         results_summary.append(result_data)

         # Mostrar resumen
         print(f"  Detectadas {len(detections)} personas")
         for i, det in enumerate(detections):
               print(f"    Persona {i+1}: Torso={det['torso_color']}, "
                     f"Piernas={det['legs_color']}, Conf={det['confidence']:.2f}")

      # Guardar resumen completo
      with open(output_dir / 'results_summary.json', 'w') as f:
         json.dump(results_summary, f, indent=2)

      print(f"\nProcesamiento completado. Resultados en: {output_dir}")
      return results_summary

# Ejemplo de uso completo
def main():
   """Función principal de demostración"""

   # Configurar rutas
   FASHIONPEDIA_ROOT = "/content/fashionpedia"  # Cambiar por tu ruta
   FASHIONPEDIA_YAML = f"{FASHIONPEDIA_ROOT}/data.yaml"
   FASHIONPEDIA_IMAGES = f"{FASHIONPEDIA_ROOT}/images/train"
   FASHIONPEDIA_LABELS = f"{FASHIONPEDIA_ROOT}/labels/train"

   PERSON_DATASET_DIR = "/content/person_dataset"
   TEST_IMAGES_DIR = "/content/test_images"
   OUTPUT_DIR = "/content/results"

   # Crear sistema
   system = PersonColorDetectionSystem()

   # Paso 1: Preparar dataset de personas desde Fashionpedia
   print("=== PASO 1: Preparando dataset de personas ===")
   dataset_config = system.prepare_person_dataset(
      FASHIONPEDIA_IMAGES,
      FASHIONPEDIA_LABELS,
      FASHIONPEDIA_YAML,
      PERSON_DATASET_DIR
   )

   # Paso 2: Entrenar YOLO para detección de personas
   print("\n=== PASO 2: Entrenando YOLO ===")
   system.train_yolo_person_detection(
      dataset_config,
      epochs=10,  # Ajustar según necesidad
      batch_size=8
   )

   # Paso 3: Cargar modelo entrenado
   print("\n=== PASO 3: Cargando modelo entrenado ===")
   # El modelo se guarda automáticamente en person_detection/yolo_person_v1/weights/best.pt
   model_path = "person_detection/yolo_person_v1/weights/best.pt"
   system.load_trained_model(model_path)

   # Paso 4: Procesar imágenes de test
   print("\n=== PASO 4: Procesando imágenes de test ===")
   results = system.batch_process_images(TEST_IMAGES_DIR, OUTPUT_DIR)

   print("\n=== PROCESO COMPLETADO ===")
   print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
   main()