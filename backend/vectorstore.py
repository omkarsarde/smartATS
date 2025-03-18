import faiss
import numpy as np
import os
import json
import traceback
import threading

# Use the embedding dimension for the sentence-transformer model
# The dimension depends on the exact model, but 384 is common for many models like 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

# Create a singleton class to manage the vector store
class VectorStore:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorStore, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the vector store with a FAISS index"""
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
        self.last_save_time = None
        self.modified_since_save = False
        self.vector_count = 0
    
    def add_vector(self, doc_id: int, embedding: list[float]) -> bool:
        """
        Add a document vector to the index with the given ID.
        Returns True if successful, False otherwise.
        """
        if doc_id is None or embedding is None:
            print(f"Warning: Skipping invalid vector (id: {doc_id}, embedding: {'None' if embedding is None else f'len={len(embedding)}'})")
            return False
            
        try:
            vector = np.array(embedding, dtype='float32')
            
            # Validate vector dimension
            if vector.shape[0] != EMBEDDING_DIM:
                print(f"Error: Vector dimension mismatch. Expected {EMBEDDING_DIM}, got {vector.shape[0]}")
                return False
                
            # Normalize the vector to unit length
            norm = np.linalg.norm(vector)
            if norm == 0:
                print(f"Warning: Zero norm vector for document {doc_id}")
                return False
                
            vector = vector / norm
            
            # Check for NaN or Inf values
            if not np.isfinite(vector).all():
                print(f"Error: Vector contains NaN or Inf values for document {doc_id}")
                return False
                
            # Add to index with document_id as the label
            self.index.add_with_ids(np.expand_dims(vector, axis=0), np.array([doc_id], dtype='int64'))
            self.modified_since_save = True
            self.vector_count += 1
            return True
        except Exception as e:
            print(f"Error adding vector: {e}")
            traceback.print_exc()
            return False
    
    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """
        Search the index for vectors most similar to the given query vector.
        Returns a list of (doc_id, similarity_score) tuples.
        """
        if self.index.ntotal == 0:
            print("Warning: Index is empty, no documents to search")
            return []
            
        try:
            print(f"Searching index with {self.index.ntotal} vectors")
            query_vector = np.array(query_vector, dtype='float32')
            
            # Validate vector dimension
            if query_vector.shape[0] != EMBEDDING_DIM:
                print(f"Error: Query vector dimension mismatch. Expected {EMBEDDING_DIM}, got {query_vector.shape[0]}")
                return []
                
            # Normalize query vector
            norm = np.linalg.norm(query_vector)
            if norm == 0:
                print("Warning: Zero norm query vector")
                return []
                
            query_vector = query_vector / norm
            
            # Check for NaN or Inf values
            if not np.isfinite(query_vector).all():
                print("Error: Query vector contains NaN or Inf values")
                return []
                
            # Perform search
            top_k = min(top_k, self.index.ntotal)  # Can't return more than we have
            distances, ids = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
            
            print(f"Search returned {len(ids[0])} results")
            
            results = []
            for i, doc_id in enumerate(ids[0]):
                if doc_id == -1:
                    continue
                score = float(distances[0][i])  # Inner product score (range -1 to 1 for cosine similarity)
                results.append((int(doc_id), score))
                
            print(f"Returning {len(results)} valid results")
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            traceback.print_exc()
            return []
    
    def save(self, file_path="faiss_index.bin") -> bool:
        """Save the index to disk"""
        try:
            if self.index.ntotal > 0:
                # Create a temporary file first to avoid corrupting the index if the save fails
                temp_file = f"{file_path}.temp"
                faiss.write_index(self.index, temp_file)
                
                # If successful, rename to the final file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                os.rename(temp_file, file_path)
                
                print(f"Saved index with {self.index.ntotal} vectors to {file_path}")
                self.modified_since_save = False
                self.last_save_time = np.datetime64('now')
                return True
            else:
                print("No vectors in index, skipping save")
                return False
        except Exception as e:
            print(f"Error saving index: {e}")
            traceback.print_exc()
            return False
    
    def load(self, file_path="faiss_index.bin") -> bool:
        """Load the index from disk"""
        try:
            if os.path.exists(file_path):
                # Load the index from disk
                loaded_index = faiss.read_index(file_path)
                
                # Verify it's a valid index with the correct dimension
                if not isinstance(loaded_index, faiss.IndexIDMap):
                    print(f"Error: Loaded index is not an IndexIDMap")
                    return False
                    
                # Check if the index has the correct dimension
                d = loaded_index.d
                if d != EMBEDDING_DIM:
                    print(f"Error: Dimension mismatch in loaded index. Expected {EMBEDDING_DIM}, got {d}")
                    return False
                
                # All checks passed, replace the index
                self.index = loaded_index
                self.vector_count = self.index.ntotal
                print(f"Loaded index with {self.index.ntotal} vectors from {file_path}")
                return True
            else:
                print(f"No existing index file found at {file_path}")
                return False
        except Exception as e:
            print(f"Error loading index: {e}")
            traceback.print_exc()
            # Ensure we have a valid index even if load failed
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
            return False
    
    def clear(self) -> None:
        """Clear the index and reset it"""
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
        self.modified_since_save = True
        self.vector_count = 0
        print("Index has been reset")
    
    def get_stats(self) -> dict:
        """Return statistics about the index"""
        try:
            return {
                "vector_count": self.index.ntotal,
                "dimension": EMBEDDING_DIM,
                "index_type": str(type(self.index)),
                "is_trained": getattr(self.index, "is_trained", True),
                "last_save_time": str(self.last_save_time) if self.last_save_time else None,
                "modified_since_save": self.modified_since_save
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {
                "vector_count": 0,
                "dimension": EMBEDDING_DIM,
                "error": str(e)
            }

# Create a global instance of the vector store
vector_store = VectorStore()

# Wrapper functions to maintain backwards compatibility
def add_resume_vector(resume_id: int, embedding: list[float]) -> bool:
    """Add a resume's embedding vector to the index"""
    return vector_store.add_vector(resume_id, embedding)

def search_similar(job_embedding: list[float], top_k: int = 5) -> list[tuple[int, float]]:
    """Search for similar resumes"""
    return vector_store.search(job_embedding, top_k)

def save_index(file_path="faiss_index.bin") -> bool:
    """Save the index to disk"""
    return vector_store.save(file_path)

def load_index(file_path="faiss_index.bin") -> bool:
    """Load the index from disk"""
    return vector_store.load(file_path)

def clear_index() -> None:
    """Clear the index"""
    vector_store.clear()

def get_index_stats() -> dict:
    """Get statistics about the index"""
    return vector_store.get_stats()