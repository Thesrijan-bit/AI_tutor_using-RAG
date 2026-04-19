import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import uuid
import shutil

load_dotenv()

class RAGEngine:
    def __init__(self, data_dir: str = "./data", persist_dir: str = "./vector_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        # Using free local embeddings since Kimi doesn't have an embedding API
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.load_existing_db()
        
    def load_and_index_documents(self, data_dir: str = "./data"):
        """
        Loads all PDFs from the data directory, chunks them, and stores them in ChromaDB.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        loader = PyPDFDirectoryLoader(self.data_dir)
        docs = loader.load()
        
        if not docs:
            return False
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        self.vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings, 
            persist_directory=self.persist_dir
        )
        return True
        
    def load_existing_db(self):
        """
        Loads the existing ChromaDB vector store.
        """
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir, 
                embedding_function=self.embeddings
            )
            return True
        return False

    def add_chat_message(self, role: str, content: str, session_id: str):
        """
        Indexes a single chat message for semantic retrieval later.
        """
        if self.vector_store is None:
            # Initialize empty DB if it doesn't exist
            self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        
        metadata = {"role": role, "session_id": session_id, "type": "chat_history"}
        self.vector_store.add_texts(texts=[content], metadatas=[metadata])
        # In modern langchain-chroma, persistence is automatic, 
        # but we ensure the object is updated.

    def query_relevant_history(self, query: str, limit: int = 10):
        """
        Searches through past chat messages for relevant context.
        Filters out redundant/duplicate queries to avoid context drowning.
        """
        if self.vector_store is None:
            return []
        
        # Fetch more than we need to allow for filtering
        results = self.vector_store.similarity_search(
            query, 
            k=limit + 10, 
            filter={"type": "chat_history"}
        )
        
        # Filter out exact or near-exact matches of the current query
        # and ensure uniqueness of snippets
        unique_results = []
        seen_content = set()
        clean_query = query.lower().strip().rstrip('?')
        
        for doc in results:
            content = doc.page_content.lower().strip().rstrip('?')
            # Skip if it's just the same question being asked again
            if content == clean_query:
                continue
            
            # Ensure we don't return the same info twice
            if content not in seen_content:
                unique_results.append(doc)
                seen_content.add(content)
            
            if len(unique_results) >= limit:
                break
                
        return unique_results

    def query_documents(self, query: str, limit: int = 3):
        """
        Searches through uploaded PDFs/documents for relevant context.
        """
        if self.vector_store is None:
            return []
        
        # Search for everything EXCEPT chat history (which are the documents)
        # Chroma's $ne (not equal) filter can be used if available, 
        # but usually, we just search without the chat_history filter for docs.
        # Here we filter for docs that DON'T have the chat_history type.
        results = self.vector_store.similarity_search(
            query, 
            k=limit, 
            filter={"type": {"$ne": "chat_history"}}
        )
        return results
