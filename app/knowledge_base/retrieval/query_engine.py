"""
Query Engine

Handles querying the knowledge base and retrieving relevant content.
Based on LangChain multimodal RAG retrieval.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# CLIP for multimodal queries
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from knowledge_base.models.document_models import (
    RetrievedChunk, RetrievedImage
)


logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine for knowledge base retrieval.
    
    Supports:
    - Text-to-text retrieval
    - Text-to-image retrieval (using CLIP)
    - Similarity search
    - Filtering by document IDs
    """
    
    def __init__(self, vector_store_manager):
        """
        Initialize query engine.
        
        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vector_store = vector_store_manager
        
        # Initialize query encoder
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.query_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.query_encoder = None
        
        # Initialize CLIP for multimodal queries
        if CLIP_AVAILABLE:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                self.clip_model.to(self.device)
                self.clip_model.eval()
            except Exception as e:
                logger.warning(f"Failed to load CLIP for queries: {e}")
                self.clip_model = None
                self.clip_processor = None
        else:
            self.clip_model = None
            self.clip_processor = None
        
        logger.info("QueryEngine initialized")
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_images: bool = True,
        filter_document_ids: Optional[List[str]] = None,
        similarity_threshold: float = 0.5
    ) -> Tuple[List[RetrievedChunk], List[RetrievedImage]]:
        """
        Query the knowledge base.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            include_images: Whether to include image results
            filter_document_ids: Filter by specific document IDs
            similarity_threshold: Minimum similarity score
            
        Returns:
            Tuple of (text_chunks, images)
        """
        logger.info(f"Querying: '{query_text}' (top_k={top_k})")
        
        # Retrieve text chunks
        text_chunks = self._retrieve_text(
            query_text, top_k, filter_document_ids, similarity_threshold
        )
        
        # Retrieve images if requested
        images = []
        if include_images and self.clip_model is not None:
            images = self._retrieve_images(
                query_text, top_k, filter_document_ids, similarity_threshold
            )
        
        logger.info(f"Retrieved {len(text_chunks)} text chunks and {len(images)} images")
        
        return text_chunks, images
    
    def _retrieve_text(
        self,
        query_text: str,
        top_k: int,
        filter_document_ids: Optional[List[str]],
        similarity_threshold: float
    ) -> List[RetrievedChunk]:
        """Retrieve relevant text chunks."""
        if self.vector_store.text_store is None:
            logger.warning("No text store available")
            return []
        
        try:
            # Generate query embedding
            if self.query_encoder:
                query_embedding = self.query_encoder.encode([query_text], convert_to_numpy=True)[0]
            else:
                logger.warning("No query encoder available")
                return []
            
            # Search in vector store
            import faiss
            
            # Normalize query
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(top_k * 2, self.vector_store.text_store.ntotal)  # Get more for filtering
            if k == 0:
                return []
            
            scores, indices = self.vector_store.text_store.search(query_embedding, k)
            
            # Convert to results
            results = []
            # Use index-to-metadata mapping for correct alignment
            index_mapping = self.vector_store.text_index_to_metadata
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(index_mapping):
                    continue
                
                # Apply threshold
                if score < similarity_threshold:
                    continue
                
                # Get metadata using proper index mapping
                metadata = index_mapping[idx]
                
                # Apply document filter
                if filter_document_ids and metadata['document_id'] not in filter_document_ids:
                    continue
                
                # Create retrieved chunk
                # Get content from metadata (now properly stored)
                content = metadata.get('content', '')
                if not content:
                    # Fallback: try to get from stored text if available
                    content = f"[Content not available - chunk_id: {metadata['chunk_id']}]"
                
                chunk = RetrievedChunk(
                    chunk_id=metadata['chunk_id'],
                    document_id=metadata['document_id'],
                    content=content,
                    page_number=metadata.get('page_number'),
                    similarity_score=float(score),
                    metadata=metadata
                )
                results.append(chunk)
                
                if len(results) >= top_k:
                    break
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving text: {e}", exc_info=True)
            return []
    
    def _retrieve_images(
        self,
        query_text: str,
        top_k: int,
        filter_document_ids: Optional[List[str]],
        similarity_threshold: float
    ) -> List[RetrievedImage]:
        """Retrieve relevant images using CLIP."""
        if self.vector_store.image_store is None or self.clip_model is None:
            logger.warning("No image store or CLIP model available")
            return []
        
        try:
            # Generate query embedding with CLIP
            inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                query_embedding = text_features.cpu().numpy()[0]
            
            # Search in vector store
            import faiss
            
            # Normalize query
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(top_k * 2, self.vector_store.image_store.ntotal)
            if k == 0:
                return []
            
            scores, indices = self.vector_store.image_store.search(query_embedding, k)
            
            # Convert to results
            results = []
            # Use index-to-metadata mapping for correct alignment
            index_mapping = self.vector_store.image_index_to_metadata
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(index_mapping):
                    continue
                
                # Apply threshold
                if score < similarity_threshold:
                    continue
                
                # Get metadata using proper index mapping
                metadata = index_mapping[idx]
                
                # Apply document filter
                if filter_document_ids and metadata['document_id'] not in filter_document_ids:
                    continue
                
                # Create retrieved image
                # Note: base64_data should be fetched from storage
                image = RetrievedImage(
                    image_id=metadata['image_id'],
                    document_id=metadata['document_id'],
                    page_number=metadata.get('page_number'),
                    caption=metadata.get('caption'),
                    base64_data="",  # Placeholder - should fetch from storage
                    similarity_score=float(score),
                    width=metadata.get('width', 0),
                    height=metadata.get('height', 0)
                )
                results.append(image)
                
                if len(results) >= top_k:
                    break
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving images: {e}", exc_info=True)
            return []
    
    def get_document_chunks(self, document_id: str) -> List[RetrievedChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of retrieved chunks
        """
        chunks = []
        for chunk_id, metadata in self.vector_store.text_metadata.items():
            if metadata['document_id'] == document_id:
                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content="",  # Placeholder
                    page_number=metadata.get('page_number'),
                    similarity_score=1.0,
                    metadata=metadata
                )
                chunks.append(chunk)
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x.metadata.get('chunk_index', 0))
        return chunks
