"""
Vector Store Manager

Manages embeddings and vector storage for multimodal RAG.
Based on LangChain multimodal approach with FAISS.
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# LangChain and embeddings
try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available")

# Alternative: Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# CLIP for image embeddings
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available for image embeddings")

from knowledge_base.models.document_models import DocumentChunk, ImageContent


logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector storage for text and image embeddings.
    
    Supports:
    - Text embeddings (OpenAI, Sentence Transformers)
    - Image embeddings (CLIP)
    - FAISS vector store
    - Metadata storage
    """
    
    def __init__(
        self,
        storage_path: str = "./knowledge_base/storage",
        embedding_model: str = "sentence-transformers"
    ):
        """
        Initialize vector store manager.
        
        Args:
            storage_path: Path to store vector indices
            embedding_model: Model type ('openai', 'sentence-transformers')
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model_type = embedding_model
        
        # Initialize text embeddings
        self.text_embeddings = self._initialize_text_embeddings(embedding_model)
        
        # Initialize image embeddings (CLIP)
        self.image_model = None
        self.image_processor = None
        if CLIP_AVAILABLE:
            try:
                self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                self.image_model.to(self.device)
                self.image_model.eval()
                logger.info(f"CLIP model loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
        
        # Vector stores
        self.text_store = None
        self.image_store = None
        
        # Metadata storage
        self.text_metadata: Dict[str, Dict] = {}
        self.image_metadata: Dict[str, Dict] = {}
        
        # Index-to-metadata mapping (for FAISS index alignment)
        self.text_index_to_metadata: List[Dict] = []
        self.image_index_to_metadata: List[Dict] = []
        
        # Load existing stores if available
        self._load_stores()
        
        logger.info(f"VectorStoreManager initialized with {embedding_model}")
    
    def _initialize_text_embeddings(self, model_type: str):
        """Initialize text embedding model."""
        if model_type == "openai" and LANGCHAIN_AVAILABLE:
            # Check for OpenAI API key
            if os.getenv("OPENAI_API_KEY"):
                return OpenAIEmbeddings()
            else:
                logger.warning("OPENAI_API_KEY not set, falling back to sentence-transformers")
                model_type = "sentence-transformers"
        
        if model_type == "sentence-transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            return SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.warning("No embedding model available, using dummy embeddings")
        return None
    
    def add_text_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add text chunks to vector store.
        
        Args:
            chunks: List of document chunks
        """
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} text chunks to vector store")
        
        # Extract texts and metadata
        texts = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "content": chunk.content,  # Store content in metadata for retrieval
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        # Create embeddings and add to store
        if LANGCHAIN_AVAILABLE and isinstance(self.text_embeddings, OpenAIEmbeddings):
            # Use LangChain FAISS
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas)
            ]
            
            if self.text_store is None:
                self.text_store = FAISS.from_documents(documents, self.text_embeddings)
            else:
                new_store = FAISS.from_documents(documents, self.text_embeddings)
                self.text_store.merge_from(new_store)
        
        elif SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self.text_embeddings, SentenceTransformer):
            # Use Sentence Transformers with manual FAISS
            embeddings = self.text_embeddings.encode(texts, convert_to_numpy=True)
            self._add_to_faiss_manual(embeddings, metadatas, store_type='text')
        
        # Save metadata with proper index mapping
        for meta in metadatas:
            self.text_metadata[meta['chunk_id']] = meta
            self.text_index_to_metadata.append(meta)
        
        # Save stores
        self._save_stores()
        
        logger.info(f"Successfully added {len(chunks)} text chunks")
    
    def add_images(self, images: List[ImageContent]) -> None:
        """
        Add images to vector store.
        
        Args:
            images: List of image contents
        """
        if not images or not CLIP_AVAILABLE:
            logger.warning("Cannot add images: CLIP not available")
            return
        
        logger.info(f"Adding {len(images)} images to vector store")
        
        import base64
        from PIL import Image
        from io import BytesIO
        
        for image_content in images:
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_content.base64_data)
                pil_image = Image.open(BytesIO(image_bytes))
                
                # Generate embedding
                inputs = self.image_processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.image_model.get_image_features(**inputs)
                    embedding = image_features.cpu().numpy()[0]
                
                # Metadata
                metadata = {
                    "image_id": image_content.image_id,
                    "document_id": image_content.document_id,
                    "page_number": image_content.page_number,
                    "image_index": image_content.image_index,
                    "caption": image_content.caption,
                    "width": image_content.width,
                    "height": image_content.height
                }
                
                # Add to FAISS
                self._add_to_faiss_manual([embedding], [metadata], store_type='image')
                
                # Save metadata with proper index mapping
                self.image_metadata[image_content.image_id] = metadata
                self.image_index_to_metadata.append(metadata)
            
            except Exception as e:
                logger.error(f"Failed to process image {image_content.image_id}: {e}")
        
        # Save stores
        self._save_stores()
        
        logger.info(f"Successfully added {len(images)} images")
    
    def _add_to_faiss_manual(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict],
        store_type: str = 'text'
    ) -> None:
        """Add embeddings to FAISS manually."""
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed")
            return
        
        # Ensure embeddings are 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Get or create index
        if store_type == 'text':
            if self.text_store is None:
                dimension = embeddings.shape[1]
                self.text_store = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            self.text_store.add(embeddings)
        else:  # image
            if self.image_store is None:
                dimension = embeddings.shape[1]
                self.image_store = faiss.IndexFlatIP(dimension)
            self.image_store.add(embeddings)
    
    def _save_stores(self) -> None:
        """Save vector stores and metadata to disk."""
        try:
            import faiss
            
            # Save text store
            if self.text_store is not None:
                if LANGCHAIN_AVAILABLE and hasattr(self.text_store, 'save_local'):
                    self.text_store.save_local(str(self.storage_path / "text_store"))
                elif isinstance(self.text_store, faiss.Index):
                    faiss.write_index(self.text_store, str(self.storage_path / "text_faiss.index"))
            
            # Save image store
            if self.image_store is not None and isinstance(self.image_store, faiss.Index):
                faiss.write_index(self.image_store, str(self.storage_path / "image_faiss.index"))
            
            # Save metadata
            with open(self.storage_path / "text_metadata.pkl", 'wb') as f:
                pickle.dump(self.text_metadata, f)
            
            with open(self.storage_path / "image_metadata.pkl", 'wb') as f:
                pickle.dump(self.image_metadata, f)
            
            # Save index mappings
            with open(self.storage_path / "text_index_mapping.pkl", 'wb') as f:
                pickle.dump(self.text_index_to_metadata, f)
            
            with open(self.storage_path / "image_index_mapping.pkl", 'wb') as f:
                pickle.dump(self.image_index_to_metadata, f)
            
            logger.info("Vector stores saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save stores: {e}")
    
    def _load_stores(self) -> None:
        """Load vector stores and metadata from disk."""
        try:
            import faiss
            
            # Load text store
            text_store_path = self.storage_path / "text_store"
            text_faiss_path = self.storage_path / "text_faiss.index"
            
            if LANGCHAIN_AVAILABLE and text_store_path.exists():
                self.text_store = FAISS.load_local(
                    str(text_store_path),
                    self.text_embeddings,
                    allow_dangerous_deserialization=True
                )
            elif text_faiss_path.exists():
                self.text_store = faiss.read_index(str(text_faiss_path))
            
            # Load image store
            image_faiss_path = self.storage_path / "image_faiss.index"
            if image_faiss_path.exists():
                self.image_store = faiss.read_index(str(image_faiss_path))
            
            # Load metadata
            text_meta_path = self.storage_path / "text_metadata.pkl"
            if text_meta_path.exists():
                with open(text_meta_path, 'rb') as f:
                    self.text_metadata = pickle.load(f)
            
            image_meta_path = self.storage_path / "image_metadata.pkl"
            if image_meta_path.exists():
                with open(image_meta_path, 'rb') as f:
                    self.image_metadata = pickle.load(f)
            
            # Load index mappings
            text_index_path = self.storage_path / "text_index_mapping.pkl"
            if text_index_path.exists():
                with open(text_index_path, 'rb') as f:
                    self.text_index_to_metadata = pickle.load(f)
            
            image_index_path = self.storage_path / "image_index_mapping.pkl"
            if image_index_path.exists():
                with open(image_index_path, 'rb') as f:
                    self.image_index_to_metadata = pickle.load(f)
            
            if self.text_store is not None or self.image_store is not None:
                logger.info("Vector stores loaded successfully")
        
        except Exception as e:
            logger.warning(f"Could not load existing stores: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector stores."""
        stats = {
            "text_chunks": len(self.text_metadata),
            "images": len(self.image_metadata),
            "text_store_size": self.text_store.ntotal if hasattr(self.text_store, 'ntotal') else 0,
            "image_store_size": self.image_store.ntotal if hasattr(self.image_store, 'ntotal') else 0
        }
        return stats
