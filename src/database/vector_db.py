import chromadb
from chromadb.config import Settings
import numpy as np

class VectorDatabase:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB for vector storage"""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = None
        self.collection_name = "media_segments"
        
    def create_or_get_collection(self):
        """Create or get the media segments collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "CLIP embeddings for media segments"}
            )
            print(f"✓ Collection '{self.collection_name}' ready")
            return self.collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def add_segments(self, embeddings, metadatas, ids):
        """
        Add segments to the vector database
        
        Args:
            embeddings: list of numpy arrays (CLIP embeddings)
            metadatas: list of dict (segment metadata)
            ids: list of str (unique segment IDs)
        """
        if self.collection is None:
            self.create_or_get_collection()
        
        embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                          for emb in embeddings]
        
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✓ Added {len(ids)} segments to vector database")
    
    def search_by_embedding(self, query_embedding, n_results=10):
        """
        Search for similar segments using an embedding vector
        
        Args:
            query_embedding: numpy array (CLIP embedding)
            n_results: number of results to return
            
        Returns:
            dict with 'ids', 'distances', 'metadatas'
        """
        if self.collection is None:
            self.create_or_get_collection()
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )
        return results
    
    def search_by_text(self, query_text, n_results=10):
        """
        Search for segments using text query
        Note: This requires encoding the text to embedding first
        This method is a placeholder - actual text encoding happens in the app
        """
        pass
    
    def get_count(self):
        """Get total number of segments in database"""
        if self.collection is None:
            self.create_or_get_collection()
        return self.collection.count()
    
    def delete_all(self):
        """Delete all segments (use with caution)"""
        if self.collection:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = None
                print("✓ All segments deleted")
                self.create_or_get_collection()
            except Exception as e:
                print(f"Error deleting collection: {e}")
