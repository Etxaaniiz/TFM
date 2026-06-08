import os
from PIL import Image

examples_dir = r"c:\Users\etxan\OneDrive\Documentos\TFM_final\output\figures\Ejemplos"
for filename in os.listdir(examples_dir):
    if filename.endswith(".png"):
        path = os.path.join(examples_dir, filename)
        try:
            with Image.open(path) as img:
                print(f"File: {filename}")
                print(f"  Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
                print(f"  Info: {img.info}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
