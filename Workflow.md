# Workflow

## 1) Convert Images to Vectors (CLIP)
- Load each image and its id from the metadata
- Preprocess per CLIP requirements
- Encode with CLIP image encoder to get a fixed-size embedding vector
- Save vectors.npy and id_index.json

## 2) Create FAISS index
- Load embeddings
- Build FAISS index
- Add embeddings to index in the same order as id_index.json
- Save faiss.index

## 3) Query FAISS
- Encode the query image with the same CLIP model
- Search FAISS for top-k neighbors
- Map FAISS result rows -> image ids via id_index.json
- Join ids with metadata
- Output the top k results
