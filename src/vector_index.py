import cv2
import faiss
import numpy as np
import os

from feature_extractor import extract_features

# Initialize FAISS index
index = faiss.IndexFlatL2(2048)  # Dimension should match feature vector size
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


def load_and_index_images(image_folder):
    metadata = []
    for object_name in os.listdir(image_folder):
        object_path = os.path.join(image_folder, object_name)
        for img_name in os.listdir(object_path):
            img_path = os.path.join(object_path, img_name)
            img = cv2.imread(img_path)
            feature_vector = extract_features(img)
            index.add(np.array([feature_vector]))
            metadata.append({
                "object_name": object_name,
                "image_path": img_path,
                "additional_info": load_metadata(img_path)
            })

def load_metadata(image_path):
    # Implement metadata loading logic here
    return {"dummy_key": "dummy_value"}