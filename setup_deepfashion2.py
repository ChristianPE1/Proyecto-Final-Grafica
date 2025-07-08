#!/usr/bin/env python3
"""
Script para descargar y preparar DeepFashion2 dataset
Contraseña: 2019Deepfashion2**
"""

import os
import zipfile
import json
from pathlib import Path

def setup_deepfashion2_colab():
    """
    Script para configurar DeepFashion2 en Google Colab
    """
    
    print("=== CONFIGURACIÓN DE DEEPFASHION2 ===")
    print("1. Descarga los archivos manualmente desde:")
    print("   https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok")
    print("   Contraseña: 2019Deepfashion2**")
    print("   Archivos necesarios: train.zip, validation.zip, json_for_validation.zip")
    print()
    
    # Crear estructura de directorios
    base_dir = Path("/content/deepfashion2")
    base_dir.mkdir(exist_ok=True)
    
    # Directorios esperados
    dirs_to_create = [
        "train/image",
        "train/annos", 
        "validation/image",
        "validation/annos"
    ]
    
    for dir_path in dirs_to_create:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    print("2. Estructura de directorios creada en /content/deepfashion2/")
    print()
    print("3. Después de descargar, ejecuta este código para extraer:")
    print()
    
    extract_code = '''
# Extraer archivos (ejecutar después de descargar)
import zipfile
from pathlib import Path

# Extraer train.zip
with zipfile.ZipFile('/content/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/deepfashion2/train/')

# Extraer validation.zip  
with zipfile.ZipFile('/content/validation.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/deepfashion2/validation/')

# Extraer json_for_validation.zip (contiene anotaciones)
with zipfile.ZipFile('/content/json_for_validation.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/deepfashion2/validation/annos/')

print("Archivos extraídos correctamente!")
'''
    
    print(extract_code)
    print()
    print("4. Una vez extraído, puedes usar el sistema de entrenamiento:")
    print("   from yolo_transfer import main_clothing_detection")
    print("   main_clothing_detection()")

def verify_deepfashion2_structure():
    """Verifica que la estructura de DeepFashion2 esté correcta"""
    
    base_dir = Path("/content/deepfashion2")
    
    required_dirs = [
        "train/image",
        "train/annos",
        "validation/image", 
        "validation/annos"
    ]
    
    print("=== VERIFICACIÓN DE ESTRUCTURA ===")
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            count = len(list(full_path.glob('*')))
            print(f"✓ {dir_path}: {count} archivos")
        else:
            print(f"✗ {dir_path}: NO EXISTE")
    
    # Verificar algunos archivos de ejemplo
    train_images = base_dir / "train/image"
    if train_images.exists():
        sample_images = list(train_images.glob('*.jpg'))[:5]
        print(f"\nEjemplos de imágenes de entrenamiento:")
        for img in sample_images:
            print(f"  - {img.name}")
    
    train_annos = base_dir / "train/annos"
    if train_annos.exists():
        sample_annos = list(train_annos.glob('*.json'))[:5]
        print(f"\nEjemplos de anotaciones:")
        for anno in sample_annos:
            print(f"  - {anno.name}")
            
            # Verificar contenido de una anotación
            try:
                with open(anno, 'r') as f:
                    data = json.load(f)
                    items = data.get('items', [])
                    print(f"    Items en {anno.name}: {len(items)}")
                    if items:
                        sample_item = items[0]
                        category_id = sample_item.get('category_id', 'N/A')
                        print(f"    Ejemplo - Categoría ID: {category_id}")
                break
            except:
                print(f"    Error al leer {anno.name}")

if __name__ == "__main__":
    setup_deepfashion2_colab()
