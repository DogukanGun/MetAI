"""
Document Models for Knowledge Base

Pydantic models for document processing and retrieval.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (pdf, docx, txt, etc.)")
    file_size: int = Field(..., description="File size in bytes")
    upload_date: datetime = Field(..., description="Upload timestamp")
    num_pages: Optional[int] = Field(None, description="Number of pages (for PDFs)")
    num_images: int = Field(0, description="Number of images extracted")
    num_text_chunks: int = Field(0, description="Number of text chunks")
    total_tokens: Optional[int] = Field(None, description="Approximate token count")


class DocumentChunk(BaseModel):
    """A chunk of document content."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index in document")
    page_number: Optional[int] = Field(None, description="Source page number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ImageContent(BaseModel):
    """Extracted image from document."""
    image_id: str = Field(..., description="Unique image identifier")
    document_id: str = Field(..., description="Parent document ID")
    page_number: Optional[int] = Field(None, description="Source page number")
    image_index: int = Field(..., description="Index in document")
    caption: Optional[str] = Field(None, description="Image caption or description")
    base64_data: str = Field(..., description="Base64 encoded image data")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class DocumentIngestionRequest(BaseModel):
    """Request for document ingestion."""
    extract_images: bool = Field(True, description="Whether to extract images")
    chunk_size: int = Field(1000, description="Text chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    generate_summaries: bool = Field(False, description="Generate chunk summaries")


class DocumentIngestionResponse(BaseModel):
    """Response after document ingestion."""
    success: bool = Field(..., description="Whether ingestion succeeded")
    message: str = Field(..., description="Status message")
    document_metadata: DocumentMetadata = Field(..., description="Document metadata")
    processing_time: float = Field(..., description="Processing time in seconds")


class QueryRequest(BaseModel):
    """Request for knowledge base query."""
    query: str = Field(..., description="Query text", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    include_images: bool = Field(True, description="Include image results")
    filter_document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    similarity_threshold: float = Field(0.5, description="Minimum similarity score", ge=0, le=1)


class RetrievedChunk(BaseModel):
    """Retrieved text chunk with similarity score."""
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Source document ID")
    content: str = Field(..., description="Chunk content")
    page_number: Optional[int] = Field(None, description="Page number")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievedImage(BaseModel):
    """Retrieved image with similarity score."""
    image_id: str = Field(..., description="Image identifier")
    document_id: str = Field(..., description="Source document ID")
    page_number: Optional[int] = Field(None, description="Page number")
    caption: Optional[str] = Field(None, description="Image caption")
    base64_data: str = Field(..., description="Base64 encoded image")
    similarity_score: float = Field(..., description="Similarity score")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")


class QueryResponse(BaseModel):
    """Response for knowledge base query."""
    success: bool = Field(..., description="Whether query succeeded")
    query: str = Field(..., description="Original query")
    retrieved_chunks: List[RetrievedChunk] = Field(..., description="Retrieved text chunks")
    retrieved_images: List[RetrievedImage] = Field(..., description="Retrieved images")
    total_results: int = Field(..., description="Total number of results")
    processing_time: float = Field(..., description="Processing time in seconds")


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    success: bool = Field(..., description="Whether request succeeded")
    documents: List[DocumentMetadata] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total document count")


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion."""
    success: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="Deleted document ID")
