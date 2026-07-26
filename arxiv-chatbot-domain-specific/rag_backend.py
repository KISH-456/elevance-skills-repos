import json
import os
from dotenv import load_dotenv  # <--- Add this
load_dotenv() 

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class ArxivRAG:
    def __init__(self, data_path, index_path="faiss_index"):
        self.data_path = data_path
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        # Ensure your OPENAI_API_KEY is set in your environment
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
 
        
    def load_and_index(self, limit=2000):
        print("Loading data...")
        documents = []
        if not os.path.exists(self.data_path):
            print(f"Error: {self.data_path} not found. Run filter_arxiv.py first.")
            return

        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                item = json.loads(line)
                text = f"Title: {item['title']}\n\nAbstract: {item['abstract']}"
                metadata = {
                    "id": item['id'],
                    "title": item['title'],
                    "authors": item['authors'],
                    "categories": item['categories']
                }
                documents.append(Document(page_content=text, metadata=metadata))
        
        print(f"Splitting {len(documents)} documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        print(f"Creating index for {len(chunks)} chunks...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.index_path)
        print("Indexing complete.")

    def search(self, query, k=5):
        if not self.vector_store:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            else:
                self.load_and_index()
        
        return self.vector_store.similarity_search(query, k=k)

    def get_answer(self, query):
        """
        Direct RAG implementation: Retrieves docs and sends them to the LLM.
        This avoids the 'langchain.chains' module entirely.
        """
        if not self.vector_store:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            else:
                self.load_and_index()
        
        # 1. Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=4)
        context = "\n\n".join([f"Source {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        
        # 2. Construct the prompt manually
        prompt = f"""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide a concise and scientific explanation.

Context:
{context}

Question: {query}
Answer:"""
        
        # 3. Get response from LLM
        response = self.llm.invoke(prompt)
        
        return {
            "result": response.content,
            "source_documents": docs
        }

if __name__ == "__main__":
    rag = ArxivRAG("arxiv_cs.jsonl")
    rag.load_and_index(limit=2000)
