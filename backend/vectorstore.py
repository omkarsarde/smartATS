import faiss
import numpy as np

# Use the embedding dimension for the sentence-transformer model
# The dimension depends on the exact model, but 768 is common for many models like 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 768

# Initialize a FAISS index for cosine similarity (inner product on normalized vectors)
index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))

def add_resume_vector(resume_id: int, embedding: list[float]):
    """
    Add a resume's embedding vector to the FAISS index with the given resume_id.
    Embedding is normalized for cosine similarity.
    """
    vector = np.array(embedding, dtype='float32')
    # Normalize the vector to unit length
    norm = np.linalg.norm(vector)
    if norm != 0:
        vector = vector / norm
    # Add to index with resume_id as the label
    index.add_with_ids(np.expand_dims(vector, axis=0), np.array([resume_id], dtype='int64'))

def search_similar(job_embedding: list[float], top_k: int = 5):
    """
    Search the FAISS index for resumes most similar to the given job description embedding.
    Returns a list of (resume_id, similarity_score).
    """
    if index.ntotal == 0:
        return []
    query_vector = np.array(job_embedding, dtype='float32')
    # Normalize query vector
    norm = np.linalg.norm(query_vector)
    if norm != 0:
        query_vector = query_vector / norm
    # Perform search
    distances, ids = index.search(np.expand_dims(query_vector, axis=0), top_k)
    results = []
    for i, res_id in enumerate(ids[0]):
        if res_id == -1:
            continue
        score = float(distances[0][i])  # Inner product score (range -1 to 1 for cosine similarity)
        results.append((int(res_id), score))
    return results