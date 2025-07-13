# -*- coding: utf-8 -*-
"""
YOLO Fashion Detection with Color Analysis
Complete pipeline for fashion detection and color extraction
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
import zipfile
import gdown
from IPython.display import Image, display
import shutil
import random
from pathlib import Path

# ==================== SETUP Y DESCARGA DE DATOS ====================

def setup_environment():
    """Configurar el entorno de Google Colab"""
    #from google.colab import drive
    #drive.mount('/content/drive')

    # Instalar dependencias
    !pip install ultralytics gdown scikit-learn --upgrade -q

    print("Entorno configurado correctamente")

def download_deepfashion2():
    """Descargar y extraer DeepFashion2 directamente en Colab usando gdown"""
    import subprocess
    import zipfile
    base_path = '/content/deepfashion2'
    os.makedirs(base_path, exist_ok=True)

    # IDs públicos de los archivos en Google Drive (asegúrate que sean públicos)
    files = {
        'train.zip': '1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK',
        'validation.zip': '1O45YqhREBOoLudjA06HcTehcEebR0o9y',
        'json_for_validation.zip': '12DmrxXNtl0U9hnN1bzue4XX7nw1fSMZ5'
    }

    zip_password = '2019Deepfashion2**'
    for fname, fileid in files.items():
        out_path = f'{base_path}/{fname}'
        if not os.path.exists(out_path):
            print(f'Descargando {fname}...')
            url = f'https://drive.google.com/uc?id={fileid}'
            try:
                subprocess.run(['gdown', '--id', fileid, '-O', out_path], check=True)
            except Exception as e:
                print(f'Error al descargar {fname}: {e}')
                continue
        else:
            print(f'{fname} ya existe, omitiendo descarga.')
        # Extraer con contraseña
        print(f'Extrayendo {fname}...')
        try:
            subprocess.run(['unzip', '-P', zip_password, out_path, '-d', base_path], check=True)
            print(f'{fname} extraído correctamente.')
        except Exception as e:
            print(f'Error al extraer {fname}: {e}')

    print('Todos los archivos procesados.')
    return True

def prepare_subset_data():
    """Preparar subset de 5k train y 1k validation"""
    base_path = '/content/deepfashion2'

    # Rutas originales
    train_images = f"{base_path}/train/image"
    train_annot = f"{base_path}/train/annot"
    val_images = f"{base_path}/validation/image"
    val_annot = f"{base_path}/validation/annot"

    # Crear directorios para subset
    subset_path = '/content/fashion_subset'
    os.makedirs(f"{subset_path}/train/images", exist_ok=True)
    os.makedirs(f"{subset_path}/train/labels", exist_ok=True)
    os.makedirs(f"{subset_path}/val/images", exist_ok=True)
    os.makedirs(f"{subset_path}/val/labels", exist_ok=True)
    os.makedirs(f"{subset_path}/test_images", exist_ok=True)

    # Seleccionar subset aleatorio
    def copy_subset(src_img, src_annot, dst_img, dst_labels, num_samples):
        all_files = [f for f in os.listdir(src_img) if f.endswith('.jpg')]
        selected_files = random.sample(all_files, min(num_samples, len(all_files)))

        for file in selected_files:
            # Copiar imagen
            shutil.copy(f"{src_img}/{file}", f"{dst_img}/{file}")

            # Copiar y convertir anotación (si existe)
            annot_file = file.replace('.jpg', '.json')
            if os.path.exists(f"{src_annot}/{annot_file}"):
                convert_deepfashion_to_yolo(f"{src_annot}/{annot_file}",
                                          f"{dst_labels}/{file.replace('.jpg', '.txt')}")

    # Procesar train y validation
    copy_subset(train_images, train_annot, f"{subset_path}/train/images",
                f"{subset_path}/train/labels", 5000)
    copy_subset(val_images, val_annot, f"{subset_path}/val/images",
                f"{subset_path}/val/labels", 1000)

    print(f"Subset creado: {len(os.listdir(f'{subset_path}/train/images'))} train, "
          f"{len(os.listdir(f'{subset_path}/val/images'))} val")

    return subset_path

def convert_deepfashion_to_yolo(json_path, txt_path):
    """Convertir anotaciones DeepFashion2 a formato YOLO"""
    import json

    # Mapeo de categorías DeepFashion2 a nuestras clases
    category_mapping = {
        1: 0,   # short sleeve top
        2: 1,   # long sleeve top
        3: 2,   # short sleeve outwear
        4: 3,   # long sleeve outwear
        5: 4,   # vest
        6: 5,   # sling
        7: 6,   # shorts
        8: 7,   # trousers
        9: 8,   # skirt
        10: 9,  # short sleeve dress
        11: 10, # long sleeve dress
        12: 11, # vest dress
        13: 12  # sling dress
    }

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        height = data['height']
        width = data['width']

        yolo_lines = []

        for item in data['items']:
            if item['category_id'] in category_mapping:
                class_id = category_mapping[item['category_id']]
                bbox = item['bounding_box']

                # Convertir a formato YOLO (normalizado)
                x_center = (bbox[0] + bbox[2]/2) / width
                y_center = (bbox[1] + bbox[3]/2) / height
                w = bbox[2] / width
                h = bbox[3] / height

                yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    except Exception as e:
        print(f"Error converting {json_path}: {e}")

def create_yaml_config(dataset_path):
    """Crear archivo de configuración YAML"""
    yaml_content = f"""
train: {dataset_path}/train/images
val: {dataset_path}/val/images

nc: 13

names:
  - short sleeve top
  - long sleeve top
  - short sleeve outwear
  - long sleeve outwear
  - vest
  - sling
  - shorts
  - trousers
  - skirt
  - short sleeve dress
  - long sleeve dress
  - vest dress
  - sling dress
"""

    yaml_path = f"{dataset_path}/data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())

    print(f"Archivo data.yaml creado en {yaml_path}")
    return yaml_path

# ==================== ANÁLISIS DE COLORES ====================

def extract_colors_kmeans(image, mask, n_colors=3):
    """Extraer colores predominantes usando K-means"""
    # Aplicar máscara
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Obtener pixeles no negros
    pixels = masked_image[mask > 0]

    if len(pixels) == 0:
        return [(0, 0, 0)] * n_colors

    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Contar frecuencia de cada color
    label_counts = Counter(labels)

    # Ordenar por frecuencia
    sorted_colors = []
    for i in sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True):
        sorted_colors.append(tuple(colors[i]))

    return sorted_colors

def rgb_to_color_name(rgb):
    """Convertir RGB a nombre de color aproximado"""
    r, g, b = rgb

    # Diccionario de colores básicos
    colors = {
        'negro': (0, 0, 0),
        'blanco': (255, 255, 255),
        'rojo': (255, 0, 0),
        'verde': (0, 255, 0),
        'azul': (0, 0, 255),
        'amarillo': (255, 255, 0),
        'naranja': (255, 165, 0),
        'rosa': (255, 192, 203),
        'morado': (128, 0, 128),
        'marron': (165, 42, 42),
        'gris': (128, 128, 128),
        'beige': (245, 245, 220),
        'navy': (0, 0, 128),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255)
    }

    min_distance = float('inf')
    closest_color = 'desconocido'

    for color_name, color_rgb in colors.items():
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, color_rgb)))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

# ==================== ENTRENAMIENTO Y EVALUACIÓN ====================

def train_yolo_model(yaml_path, epochs=30):
    """Entrenar modelo YOLO con seguimiento de métricas"""
    from ultralytics import YOLO

    # Cargar modelo base
    model = YOLO("yolov8n.pt")

    # Entrenar modelo
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        freeze=12,
        patience=10,
        save=True,
        plots=True
    )

    return model, results

def plot_training_metrics():
    """Visualizar métricas de entrenamiento"""
    results_path = "/content/runs/detect/train"

    # Leer métricas
    if os.path.exists(f"{results_path}/results.csv"):
        df = pd.read_csv(f"{results_path}/results.csv")
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
        axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Validation Loss
        axes[0, 1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
        axes[0, 1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision and Recall
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # mAP
        axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[1, 1].set_title('Mean Average Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('/content/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Mostrar otras gráficas generadas por YOLO
    plots_path = f"{results_path}"
    plot_files = ['confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'results.png']

    for plot_file in plot_files:
        if os.path.exists(f"{plots_path}/{plot_file}"):
            print(f"\n{plot_file}:")
            display(Image(f"{plots_path}/{plot_file}"))

def test_model_with_colors(model_path, test_images_path):
    """Probar modelo con imágenes de test y análisis de colores"""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Crear directorio de resultados
    results_dir = "/content/test_results"
    os.makedirs(results_dir, exist_ok=True)

    results_data = []

    for image_file in os.listdir(test_images_path):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = f"{test_images_path}/{image_file}"

            # Predicción
            results = model.predict(source=image_path, conf=0.4, save=False)

            # Cargar imagen original
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Analizar cada detección
            for i, result in enumerate(results):
                boxes = result.boxes
                if boxes is not None:
                    for j, box in enumerate(boxes):
                        # Información de la detección
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]

                        # Coordenadas del bounding box
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                        # Crear máscara para la región
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        mask[y1:y2, x1:x2] = 255

                        # Extraer colores
                        colors = extract_colors_kmeans(image, mask, n_colors=3)
                        color_names = [rgb_to_color_name(color) for color in colors]

                        # Guardar resultado
                        result_data = {
                            'image': image_file,
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'colors_rgb': colors,
                            'colors_names': color_names
                        }
                        results_data.append(result_data)

                        # Dibujar en imagen
                        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Texto con clase y colores
                        text = f"{class_name} ({confidence:.2f})"
                        cv2.putText(image_rgb, text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Mostrar colores
                        color_text = f"Colores: {', '.join(color_names[:2])}"
                        cv2.putText(image_rgb, color_text, (x1, y2+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Guardar imagen con resultados
            output_path = f"{results_dir}/result_{image_file}"
            cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            # Mostrar resultado
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title(f"Detección en {image_file}")
            plt.show()

    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(f"{results_dir}/detection_results.csv", index=False)

    print(f"\nResultados guardados en {results_dir}")
    print(f"Total de detecciones: {len(results_data)}")

    return df_results

# ==================== PIPELINE PRINCIPAL ====================

def main():
    """Pipeline principal"""
    print("=== YOLO Fashion Detection with Color Analysis ===\n")

    # 1. Configurar entorno
    print("1. Configurando entorno...")
    setup_environment()

    # 2. Descargar dataset
    print("\n2. Descargando dataset DeepFashion2...")
    if not download_deepfashion2():
        print("Error: No se pudo descargar el dataset")
        return

    # 3. Preparar subset
    print("\n3. Preparando subset de datos...")
    dataset_path = prepare_subset_data()

    # 4. Crear configuración YAML
    print("\n4. Creando configuración...")
    yaml_path = create_yaml_config(dataset_path)

    # 5. Entrenar modelo
    print("\n5. Entrenando modelo YOLO...")
    model, results = train_yolo_model(yaml_path, epochs=30)

    # 6. Visualizar métricas
    print("\n6. Visualizando métricas de entrenamiento...")
    plot_training_metrics()

    # 7. Probar modelo
    print("\n7. Probando modelo con imágenes de test...")
    model_path = "/content/runs/detect/train/weights/best.pt"
    test_path = "/content/fashion_subset/test_images"

    if os.path.exists(test_path) and os.listdir(test_path):
        results_df = test_model_with_colors(model_path, test_path)
        print("\nResultados del test:")
        print(results_df.head())
    else:
        print("No hay imágenes en test_images. Sube imágenes a esa carpeta para probar.")

    print("\n=== Pipeline completado ===")

# ==================== FUNCIONES AUXILIARES ====================

def upload_test_images():
    """Subir imágenes de test"""
    from google.colab import files

    print("Sube las imágenes de test:")
    uploaded = files.upload()

    # Mover archivos a carpeta test_images
    test_dir = "/content/fashion_subset/test_images"
    os.makedirs(test_dir, exist_ok=True)

    for filename in uploaded.keys():
        shutil.move(filename, f"{test_dir}/{filename}")

    print(f"Imágenes subidas a {test_dir}")

def analyze_single_image(image_path, model_path):
    """Analizar una sola imagen con el modelo entrenado"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.4, save=True)

    # Mostrar resultado
    for result in results:
        result.show()

        # Análisis detallado
        if result.boxes is not None:
            print(f"\nDetecciones en {image_path}:")
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"  {i+1}. {class_name} (confianza: {confidence:.3f})")

if __name__ == "__main__":
    main()