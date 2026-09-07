import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class KnowledgeBaseManager:
    """
    Manages a FAISS vector database for dynamic knowledge base expansion.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', db_path='/home/ubuntu/knowledge_base.index', metadata_path='/home/ubuntu/metadata.pkl'):
        print(f"Initializing Knowledge Base Manager with {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Load or create index
        if os.path.exists(self.db_path):
            print("Loading existing vector database...")
            self.index = faiss.read_index(self.db_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            print("Creating new vector database...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def add_information(self, texts):
        """
        Adds new information (texts) to the vector database.
        """
        if not texts:
            return
        
        print(f"Adding {len(texts)} new entries to the knowledge base...")
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata.extend(texts)
        
        # Save updates
        faiss.write_index(self.index, self.db_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("Knowledge base updated and saved.")

    def search(self, query, top_k=3):
        """
        Searches the vector database for the most relevant information.
        """
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

if __name__ == "__main__":
    # Quick test
    kb = KnowledgeBaseManager()
    kb.add_information([
        "The capital of France is Paris.",
        "The current CEO of the company is Jane Doe.",
        "The project deadline is December 31st, 2026."
    ])
    
    query = "Who is the CEO?"
    print(f"Query: {query}")
    results = kb.search(query)
    print(f"Results: {results}")
