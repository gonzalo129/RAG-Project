import faiss
import numpy as np
import os

# === CONFIGURATION ===
vector_file = r"C:\Repo\RAG Project\Embeddings\vectors.npy"
output_index_file = r"C:\Repo\RAG Project\Embeddings\valve_index.faiss"

# === LOAD VECTORS ===
vectors = np.load(vector_file).astype("float32")

# === NORMALIZE VECTORS (for cosine similarity)
faiss.normalize_L2(vectors)

# === CREATE FAISS INDEX
index = faiss.IndexFlatIP(vectors.shape[1])  # IP = inner product ≈ cosine
index.add(vectors)

print(f" FAISS index created with {index.ntotal} vectors.")

# === SAVE INDEX TO FILE
faiss.write_index(index, output_index_file)
print(f" Index saved to: {output_index_file}")
