"""
Document Processor

Handles document ingestion: loading, parsing, chunking, and embedding.
Based on LangChain multimodal RAG approach.
"""

import os
import uuid
import base64
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from io import BytesIO

# PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available, PDF processing will be limited")

from PIL import Image

# LangChain imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available, using basic text splitting")

from knowledge_base.models.document_models import (
    DocumentMetadata, DocumentChunk, ImageContent
)


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents for knowledge base ingestion.
    
    Supports:
    - PDF files with text and image extraction
    - Text files
    - DOCX files
    - Image files
    """
    
    def __init__(
        self,
        storage_path: str = "./knowledge_base/storage",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document processor.
        
        Args:
            storage_path: Path to store extracted content
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            self.text_splitter = None
        
        logger.info(f"DocumentProcessor initialized with storage at {storage_path}")
    
    def process_document(
        self,
        file_path: str,
        extract_images: bool = True,
        generate_summaries: bool = False
    ) -> Tuple[DocumentMetadata, List[DocumentChunk], List[ImageContent]]:
        """
        Process a document and extract content.
        
        Args:
            file_path: Path to document file
            extract_images: Whether to extract images
            generate_summaries: Whether to generate summaries
            
        Returns:
            Tuple of (metadata, text_chunks, images)
        """
        logger.info(f"Processing document: {file_path}")
        start_time = time.time()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Determine file type
        file_ext = file_path.suffix.lower()
        file_size = file_path.stat().st_size
        
        # Extract content based on file type
        if file_ext == '.pdf':
            text, images, num_pages = self._process_pdf(
                file_path, document_id, extract_images
            )
        elif file_ext in ['.txt', '.md']:
            text = file_path.read_text(encoding='utf-8')
            images = []
            num_pages = None
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            text = ""
            images = [self._process_image_file(file_path, document_id, 0)]
            num_pages = None
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Chunk text
        text_chunks = self._chunk_text(text, document_id)
        
        # Calculate token count (approximate)
        total_tokens = len(text.split()) if text else 0
        
        # Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file_path.name,
            file_type=file_ext[1:],  # Remove leading dot
            file_size=file_size,
            upload_date=datetime.now(),
            num_pages=num_pages,
            num_images=len(images),
            num_text_chunks=len(text_chunks),
            total_tokens=total_tokens
        )
        
        processing_time = time.time() - start_time
        logger.info(
            f"Processed {file_path.name}: {len(text_chunks)} chunks, "
            f"{len(images)} images in {processing_time:.2f}s"
        )
        
        return metadata, text_chunks, images
    
    def _process_pdf(
        self,
        pdf_path: Path,
        document_id: str,
        extract_images: bool
    ) -> Tuple[str, List[ImageContent], int]:
        """
        Process PDF file.
        
        Args:
            pdf_path: Path to PDF
            document_id: Document identifier
            extract_images: Whether to extract images
            
        Returns:
            Tuple of (text, images, num_pages)
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF processing")
        
        doc = fitz.open(pdf_path)
        text_parts = []
        images = []
        
        for page_num, page in enumerate(doc, start=1):
            # Extract text
            page_text = page.get_text()
            text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
            
            # Extract images if requested
            if extract_images:
                image_list = page.get_images()
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        pil_image = Image.open(BytesIO(image_bytes))
                        
                        # Resize if too large
                        max_size = 1024
                        if max(pil_image.size) > max_size:
                            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffered = BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Create image content
                        image_content = ImageContent(
                            image_id=f"{document_id}_img_{page_num}_{img_index}",
                            document_id=document_id,
                            page_number=page_num,
                            image_index=len(images),
                            caption=f"Image from page {page_num}",
                            base64_data=img_base64,
                            width=pil_image.width,
                            height=pil_image.height
                        )
                        images.append(image_content)
                    
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
        
        # Get page count before closing document
        num_pages = len(doc)
        doc.close()
        full_text = "\n".join(text_parts)
        
        return full_text, images, num_pages
    
    def _process_image_file(
        self,
        image_path: Path,
        document_id: str,
        image_index: int
    ) -> ImageContent:
        """
        Process standalone image file.
        
        Args:
            image_path: Path to image
            document_id: Document identifier
            image_index: Image index
            
        Returns:
            ImageContent object
        """
        pil_image = Image.open(image_path)
        
        # Resize if too large
        max_size = 1024
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return ImageContent(
            image_id=f"{document_id}_img_{image_index}",
            document_id=document_id,
            page_number=None,
            image_index=image_index,
            caption=image_path.name,
            base64_data=img_base64,
            width=pil_image.width,
            height=pil_image.height
        )
    
    def _chunk_text(
        self,
        text: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Use LangChain text splitter if available
        if self.text_splitter and LANGCHAIN_AVAILABLE:
            chunks = self.text_splitter.split_text(text)
        else:
            # Simple chunking fallback
            chunks = self._simple_chunk(text)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                content=chunk_text.strip(),
                chunk_index=i,
                page_number=None,  # Could be extracted from page markers
                metadata={"length": len(chunk_text)}
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple text chunking fallback."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep last few words for overlap
                overlap_words = int(self.chunk_overlap / 10)  # Rough estimate
                current_chunk = current_chunk[-overlap_words:]
                current_length = sum(len(w) + 1 for w in current_chunk)
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
