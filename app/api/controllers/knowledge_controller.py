"""
Knowledge Base Controller

Handles HTTP endpoints for document management and querying.
"""

import time
import tempfile
import os
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import Optional

from knowledge_base.models.document_models import (
    DocumentIngestionRequest, DocumentIngestionResponse,
    QueryRequest, QueryResponse,
    DocumentListResponse, DocumentDeleteResponse
)
from knowledge_base.ingestion.document_processor import DocumentProcessor
from knowledge_base.ingestion.vector_store_manager import VectorStoreManager
from knowledge_base.retrieval.query_engine import QueryEngine


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base"])

# Get storage path relative to app directory
_APP_DIR = Path(__file__).parent.parent.parent  # Go up to app/
_STORAGE_PATH = str(_APP_DIR / "knowledge_base" / "storage")

# Initialize components (singleton pattern)
_vector_store: Optional[VectorStoreManager] = None
_query_engine: Optional[QueryEngine] = None


def get_vector_store() -> VectorStoreManager:
    """Get or create vector store manager."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager(
            storage_path=_STORAGE_PATH,
            embedding_model="sentence-transformers"
        )
    return _vector_store


def get_query_engine() -> QueryEngine:
    """Get or create query engine."""
    global _query_engine
    if _query_engine is None:
        vector_store = get_vector_store()
        _query_engine = QueryEngine(vector_store)
    return _query_engine


@router.post(
    "/documents",
    response_model=DocumentIngestionResponse,
    summary="Upload and process a document",
    description="Upload a document (PDF, TXT, DOCX, or image) to add to the knowledge base"
)
async def upload_document(
    document: UploadFile = File(..., description="Document file to process"),
    extract_images: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    generate_summaries: bool = False
) -> DocumentIngestionResponse:
    """
    Upload and process a document for the knowledge base.
    
    Args:
        document: Document file
        extract_images: Whether to extract images from document
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        generate_summaries: Generate summaries for chunks
        
    Returns:
        DocumentIngestionResponse with metadata
    """
    start_time = time.time()
    temp_path = None
    
    try:
        logger.info(f"Received document upload: {document.filename}")
        
        # Validate file type
        allowed_extensions = ['.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.docx']
        file_ext = os.path.splitext(document.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": f"Unsupported file type: {file_ext}",
                    "details": {"allowed_extensions": allowed_extensions}
                }
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as tmp_file:
            content = await document.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Process document
        processor = DocumentProcessor(
            storage_path=_STORAGE_PATH,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        metadata, text_chunks, images = processor.process_document(
            file_path=temp_path,
            extract_images=extract_images,
            generate_summaries=generate_summaries
        )
        
        # Add to vector store
        vector_store = get_vector_store()
        vector_store.add_text_chunks(text_chunks)
        vector_store.add_images(images)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Successfully processed {document.filename}: "
            f"{len(text_chunks)} chunks, {len(images)} images in {processing_time:.2f}s"
        )
        
        return DocumentIngestionResponse(
            success=True,
            message=f"Successfully processed document with {len(text_chunks)} chunks and {len(images)} images",
            document_metadata=metadata,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "ProcessingError",
                "message": f"Failed to process document: {str(e)}",
                "details": {"filename": document.filename}
            }
        )
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description="Search for relevant content in the knowledge base"
)
async def query_knowledge_base(
    request: QueryRequest = Body(...)
) -> QueryResponse:
    """
    Query the knowledge base.
    
    Args:
        request: Query request with parameters
        
    Returns:
        QueryResponse with retrieved content
    """
    start_time = time.time()
    
    try:
        logger.info(f"Querying knowledge base: '{request.query}'")
        
        # Get query engine
        query_engine = get_query_engine()
        
        # Execute query
        text_chunks, images = query_engine.query(
            query_text=request.query,
            top_k=request.top_k,
            include_images=request.include_images,
            filter_document_ids=request.filter_document_ids,
            similarity_threshold=request.similarity_threshold
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Query returned {len(text_chunks)} text chunks and "
            f"{len(images)} images in {processing_time:.2f}s"
        )
        
        return QueryResponse(
            success=True,
            query=request.query,
            retrieved_chunks=text_chunks,
            retrieved_images=images,
            total_results=len(text_chunks) + len(images),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "QueryError",
                "message": f"Failed to query knowledge base: {str(e)}",
                "details": {"query": request.query}
            }
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="Get a list of all documents in the knowledge base"
)
async def list_documents() -> DocumentListResponse:
    """
    List all documents in the knowledge base.
    
    Returns:
        DocumentListResponse with document list
    """
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        
        # Extract unique document IDs from metadata
        document_ids = set()
        for metadata in vector_store.text_metadata.values():
            document_ids.add(metadata['document_id'])
        for metadata in vector_store.image_metadata.values():
            document_ids.add(metadata['document_id'])
        
        # For now, return basic stats
        # In a full implementation, you'd store and retrieve full document metadata
        logger.info(f"Found {len(document_ids)} documents in knowledge base")
        
        return DocumentListResponse(
            success=True,
            documents=[],  # Would populate with actual document metadata
            total_count=len(document_ids)
        )
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "ListError",
                "message": f"Failed to list documents: {str(e)}"
            }
        )


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Remove a document and its content from the knowledge base"
)
async def delete_document(document_id: str) -> DocumentDeleteResponse:
    """
    Delete a document from the knowledge base.
    
    Args:
        document_id: Document identifier
        
    Returns:
        DocumentDeleteResponse with deletion status
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # Note: Full deletion would require rebuilding the FAISS index
        # For now, we'll just acknowledge the request
        # In production, you'd implement proper deletion logic
        
        return DocumentDeleteResponse(
            success=True,
            message=f"Document deletion not fully implemented. Would delete: {document_id}",
            document_id=document_id
        )
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "DeleteError",
                "message": f"Failed to delete document: {str(e)}",
                "details": {"document_id": document_id}
            }
        )


@router.get(
    "/stats",
    summary="Get knowledge base statistics",
    description="Get statistics about the knowledge base"
)
async def get_stats():
    """
    Get knowledge base statistics.
    
    Returns:
        Statistics dictionary
    """
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "StatsError",
                "message": f"Failed to get stats: {str(e)}"
            }
        )
