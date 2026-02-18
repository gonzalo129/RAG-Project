import os
import json
import torch
import numpy as np
from PIL import Image
import faiss
from transformers import CLIPProcessor, CLIPModel

# === CONFIGURATION ===
query_image_path = r"C:\Repo\RAG Project\TestValve.jpg"
vector_file = r"C:\Repo\RAG Project\Embeddings\vectors.npy"
faiss_index_file = r"C:\Repo\RAG Project\Embeddings\valve_index.faiss"
id_index_file = r"C:\Repo\RAG Project\Embeddings\id_index.json"
model_name = "openai/clip-vit-base-patch32"
top_k = 5

# === LOAD CLIP MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# === LOAD FAISS INDEX ===
index = faiss.read_index(faiss_index_file)

# === LOAD ID INDEX (to map back to filenames)
with open(id_index_file, "r") as f:
    id_index = json.load(f)

# === PROCESS NEW IMAGE
image = Image.open(query_image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    features = model.get_image_features(**inputs)

# Normalize to unit vector
query_vector = features / features.norm(p=2, dim=-1, keepdim=True)
query_vector = query_vector.cpu().numpy().astype("float32")

# === SEARCH FAISS INDEX
faiss.normalize_L2(query_vector)  # Normalize query vector (cosine)
D, I = index.search(query_vector, k=top_k)

# === SHOW RESULTS
print(f"\n Top {top_k} similar images to '{os.path.basename(query_image_path)}':\n")
for rank, (idx, score) in enumerate(zip(I[0], D[0])):
    filename = id_index.get(str(idx), f"[Missing index {idx}]")
    print(f"{rank+1}. {filename} (similarity: {score:.4f})")
