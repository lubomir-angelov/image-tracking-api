import cv2
import faiss
import numpy as np
import pandas as pd
import os
from PIL import Image
from typing import List, Dict

from feature_extractor import extract_features

# Initialize FAISS index
index = faiss.IndexFlatL2(1000)  # Dimension should match feature vector size
metadata_store = {}  # Store metadata with vectors

def add_to_index(image, metadata):
    features = extract_features(image)
    index.add(np.array([features]))
    metadata_store[len(metadata_store)] = metadata

def search_index(query_image):
    query_features = extract_features(query_image)
    D, I = index.search(np.array([query_features]), k=5)  # Search top 5
    results = [metadata_store[i] for i in I[0]]
    return results


def load_and_index_images(image_folder: str, csv_metadata: dict) -> List:
    metadata = []
    for object_name in os.listdir(image_folder):
        object_path = os.path.join(image_folder, object_name)
        for img_name in os.listdir(object_path):
            img_path = os.path.join(object_path, img_name)
            # images have extra dimension RGBA
            # https://stackoverflow.com/questions/58496858/pytorch-runtimeerror-the-size-of-tensor-a-4-must-match-the-size-of-tensor-b
            # convert to RGB to resolve
            img = Image.open(img_path).convert('RGB')
            feature_vector = extract_features(img)
            index.add(np.array(feature_vector).reshape(1, -1))
            metadata.append({
                "object_name": object_name,
                "image_path": img_path,
                "additional_info": csv_metadata[int(object_name)]
            })
    
    return metadata


def load_metadata(csv_path: str) -> Dict:
    metadata_dict = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        gtin_id = row['GTIN_ID']
        metadata_dict[gtin_id] = row.to_dict()
        # Remove GTN_ID from the dictionary to avoid redundancy
        metadata_dict[gtin_id].pop('GTIN_ID', None)

    return metadata_dict