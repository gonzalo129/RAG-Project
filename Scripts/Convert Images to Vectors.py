import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === CONFIGURATION ===
image_folder = r"C:\Repo\RAG Project\Images"  # Update this!
output_folder = r"C:\Repo\Rag Project\Embeddings"
output_vector_file = "vectors.npy"
output_id_index_file = "id_index.json"
model_name = "openai/clip-vit-base-patch32"

# === LOAD CLIP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# === LOAD AND EMBED IMAGES ===
vectors = []
id_index = {}

# Only use .png files that start with "Valve"
image_files = sorted(f for f in os.listdir(image_folder) if f.startswith("Valve") and f.endswith(".png"))

for idx, filename in tqdm(enumerate(image_files), total=len(image_files)):
    path = os.path.join(image_folder, filename)
    image = Image.open(path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # Normalize the feature vector to unit length (for cosine similarity)
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)

    vectors.append(normalized.squeeze().cpu().numpy())
    id_index[idx] = filename

# === SAVE TO FILES ===
np.save(output_vector_file, np.stack(vectors))

with open(output_id_index_file, "w") as f:
    json.dump(id_index, f, indent=2)

print(f" Embedded {len(vectors)} images.")
print(f" Saved vectors to: {output_vector_file}")
print(f" Saved index map to: {output_id_index_file}")
