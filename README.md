# RAG-Project
Image retrieval (RAG-Style) workflow built on synthetic data

**Goal:** Showcase the workflow

## What it does
1) Convert images to embeddings (**CLIP Vectors**)
2) Build a vector database (**FAISS index**) out of synthetic data storing vectors
3) Query FAISS with a new image vector to retreive Top-k similar images, then inspect and compare

## Workflow

### 1) Convert images to vectors
Script: Scripts/Convert Images to Vectors.py

**Inputs**
- Images/ folder
- MaterialMetadata / folder
  
**Outputs**
- Outputs/id_index
- Outputs/vectors.npy

### 2) Create FAISS database
Script: Scripts/Create Database.py

**Inputs**
-embeddings + ids

**Outputs**
- Outputs/valves_index.faiss

### 3) Query FAISS
Script: Scripts/ Query FAISS.py

**Inputs**
- Query Image (Tests/test image.png)
- FAISS Index

**Outputs**
Top k results, with k being specified in Query FAISS.py

## Dataset
The full synthetic data set (1000 images) is not included in this repo.

## Notes
- CLIP embeddings + FAISS provide fast similarity search for asset identification
- Metadata validation is a useful sanity check that retrieval is semantically consistent (ie same color, material etc)


