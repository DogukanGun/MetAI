# Knowledge Base System

Multimodal RAG (Retrieval-Augmented Generation) system for document management and intelligent retrieval. Based on LangChain multimodal approach with FAISS vector storage.

## Overview

This knowledge base system provides:
- **Document Ingestion**: Process and store PDFs, text files, DOCX, and images
- **Multimodal RAG**: Retrieve both text chunks and images using semantic similarity
- **Vector Storage**: FAISS-based efficient similarity search
- **Separate Pipelines**: Independent document processing and querying

## Architecture

```
knowledge_base/
├── ingestion/                 # Document processing and storage
│   ├── document_processor.py  # Parse and chunk documents
│   └── vector_store_manager.py# Manage embeddings and FAISS
├── retrieval/                 # Query and search
│   └── query_engine.py        # Similarity search and retrieval
├── models/                    # Pydantic data models
│   └── document_models.py     # Request/response models
├── storage/                   # Vector indices and metadata
│   ├── text_faiss.index       # Text embeddings
│   ├── image_faiss.index      # Image embeddings (CLIP)
│   ├── text_metadata.pkl      # Text chunk metadata
│   └── image_metadata.pkl     # Image metadata
└── utils/                     # Utility functions
```

## Features

### Document Ingestion
- **PDF Processing**: Extract text and images using PyMuPDF
- **Text Chunking**: Intelligent text splitting with LangChain RecursiveCharacterTextSplitter
- **Image Extraction**: Extract and resize images from PDFs
- **Metadata Tracking**: Store document metadata, page numbers, timestamps

### Embeddings
- **Text Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) or OpenAI embeddings
- **Image Embeddings**: CLIP (openai/clip-vit-base-patch32) for multimodal retrieval
- **Vector Storage**: FAISS with cosine similarity (Inner Product after normalization)

### Retrieval
- **Text-to-Text**: Semantic search for relevant text chunks
- **Text-to-Image**: Find images related to text queries using CLIP
- **Filtering**: Filter results by document IDs
- **Thresholding**: Minimum similarity score filtering

## API Endpoints

### 1. Upload Document

**POST** `/api/v1/knowledge/documents`

Upload and process a document for the knowledge base.

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "document=@path/to/document.pdf" \
  -F "extract_images=true" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed document with 45 chunks and 3 images",
  "document_metadata": {
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "document.pdf",
    "file_type": "pdf",
    "file_size": 1234567,
    "upload_date": "2025-10-10T00:00:00",
    "num_pages": 10,
    "num_images": 3,
    "num_text_chunks": 45,
    "total_tokens": 3500
  },
  "processing_time": 12.5
}
```

### 2. Query Knowledge Base

**POST** `/api/v1/knowledge/query`

Search for relevant content in the knowledge base.

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "top_k": 5,
    "include_images": true,
    "similarity_threshold": 0.5
  }'
```

**Response:**
```json
{
  "success": true,
  "query": "What are the main findings?",
  "retrieved_chunks": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000_chunk_5",
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "The main findings indicate...",
      "page_number": 3,
      "similarity_score": 0.87,
      "metadata": {...}
    }
  ],
  "retrieved_images": [
    {
      "image_id": "550e8400-e29b-41d4-a716-446655440000_img_3_0",
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "page_number": 3,
      "caption": "Image from page 3",
      "base64_data": "iVBORw0KGgoAAAANS...",
      "similarity_score": 0.75,
      "width": 800,
      "height": 600
    }
  ],
  "total_results": 8,
  "processing_time": 0.5
}
```

### 3. List Documents

**GET** `/api/v1/knowledge/documents`

Get a list of all documents in the knowledge base.

```bash
curl "http://localhost:8000/api/v1/knowledge/documents"
```

### 4. Delete Document

**DELETE** `/api/v1/knowledge/documents/{document_id}`

Remove a document from the knowledge base.

```bash
curl -X DELETE "http://localhost:8000/api/v1/knowledge/documents/{document_id}"
```

### 5. Get Statistics

**GET** `/api/v1/knowledge/stats`

Get knowledge base statistics.

```bash
curl "http://localhost:8000/api/v1/knowledge/stats"
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "text_chunks": 150,
    "images": 12,
    "text_store_size": 150,
    "image_store_size": 12
  }
}
```

## Programmatic Usage

### Document Ingestion

```python
from knowledge_base.ingestion.document_processor import DocumentProcessor
from knowledge_base.ingestion.vector_store_manager import VectorStoreManager

# Initialize processor
processor = DocumentProcessor(
    storage_path="./knowledge_base/storage",
    chunk_size=1000,
    chunk_overlap=200
)

# Process document
metadata, text_chunks, images = processor.process_document(
    file_path="path/to/document.pdf",
    extract_images=True
)

# Add to vector store
vector_store = VectorStoreManager(
    storage_path="./knowledge_base/storage",
    embedding_model="sentence-transformers"
)
vector_store.add_text_chunks(text_chunks)
vector_store.add_images(images)
```

### Querying

```python
from knowledge_base.retrieval.query_engine import QueryEngine

# Initialize query engine
query_engine = QueryEngine(vector_store)

# Query
text_chunks, images = query_engine.query(
    query_text="What are the main findings?",
    top_k=5,
    include_images=True,
    similarity_threshold=0.5
)

# Process results
for chunk in text_chunks:
    print(f"Score: {chunk.similarity_score:.2f} - {chunk.content[:100]}...")
```

## Configuration

### Embedding Models

**Sentence Transformers** (Default):
- Model: `all-MiniLM-L6-v2`
- Dimension: 384
- Fast and efficient
- No API key required

**OpenAI Embeddings**:
- Requires `OPENAI_API_KEY` environment variable
- Model: `text-embedding-ada-002`
- Dimension: 1536
- High quality embeddings

### CLIP for Images

- Model: `openai/clip-vit-base-patch32`
- Dimension: 512
- Multimodal text-image embeddings

### Text Chunking

- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)
- **Splitter**: RecursiveCharacterTextSplitter (preserves semantic boundaries)

## Supported File Types

- **PDF**: `.pdf` - Text and image extraction
- **Text**: `.txt`, `.md` - Plain text
- **Images**: `.png`, `.jpg`, `.jpeg` - Image-only documents
- **DOCX**: `.docx` - Word documents (requires python-docx)

## Storage

### Vector Indices

- **Text Index**: `storage/text_faiss.index` - FAISS IndexFlatIP (cosine similarity)
- **Image Index**: `storage/image_faiss.index` - FAISS IndexFlatIP (cosine similarity)

### Metadata

- **Text Metadata**: `storage/text_metadata.pkl` - Chunk metadata (chunk_id, document_id, page_number, etc.)
- **Image Metadata**: `storage/image_metadata.pkl` - Image metadata (image_id, document_id, caption, dimensions)

## Performance

### Document Processing

- **PDF (10 pages)**: ~5-10 seconds
- **PDF (100 pages)**: ~30-60 seconds
- Image extraction adds ~1-2 seconds per image

### Querying

- **Text search**: ~50-200ms for 1000 chunks
- **Image search**: ~100-300ms for 100 images
- FAISS is highly optimized for large-scale similarity search

## Dependencies

```
langchain>=0.1.0
langchain-community>=0.0.20
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
PyMuPDF>=1.23.0
Pillow>=10.0.0
transformers>=4.35.0
torch>=2.0.0
```

Install with:
```bash
pip install -r requirements_knowledge.txt
```

## Limitations

### Current Implementation

1. **Document Deletion**: Not fully implemented (requires FAISS index rebuilding)
2. **Content Storage**: Document content not stored with metadata (only in chunks)
3. **Update**: No document update mechanism (delete and re-upload)
4. **Persistence**: Metadata stored in pickle files (consider database for production)

### Scalability

- FAISS IndexFlatIP: Good for up to ~1M vectors
- For larger scales: Consider IndexIVFFlat or IndexHNSWFlat
- For distributed: Consider Pinecone, Weaviate, or Qdrant

## Best Practices

### Document Preparation

1. **Clean PDFs**: Ensure PDFs have extractable text (not scanned images)
2. **Optimal Size**: 1-50 pages per document for best performance
3. **Image Quality**: High-resolution images work better with CLIP

### Chunking Strategy

1. **Semantic Boundaries**: RecursiveCharacterTextSplitter respects paragraphs
2. **Chunk Size**: 500-1500 characters works well
3. **Overlap**: 10-20% overlap helps maintain context

### Query Optimization

1. **Specific Queries**: More specific queries return better results
2. **Threshold**: Adjust similarity threshold based on precision/recall needs
3. **Top-K**: Start with 5-10 results, adjust as needed

## Integration with LLMs

The knowledge base is designed to work seamlessly with LLMs:

```python
# Retrieve context
chunks, images = query_engine.query("What is X?", top_k=3)

# Build context for LLM
context = "\n\n".join([chunk.content for chunk in chunks])

# Send to LLM
prompt = f"""
Context from knowledge base:
{context}

Question: What is X?

Answer based on the context above:
"""

# Use with OpenAI, Anthropic, etc.
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'knowledge_base'`:
- Ensure you're running from the `app/` directory
- Or add `app/` to PYTHONPATH: `export PYTHONPATH=/path/to/app:$PYTHONPATH`

### FAISS Not Found

Install FAISS:
```bash
pip install faiss-cpu  # For CPU
pip install faiss-gpu  # For GPU (requires CUDA)
```

### CLIP Out of Memory

Reduce image size or batch size:
- Images are automatically resized to max 1024px
- Process images in smaller batches

## Future Enhancements

- [ ] Implement proper document deletion
- [ ] Add hybrid search (keyword + semantic)
- [ ] Support more document types (HTML, CSV, etc.)
- [ ] Add document summaries and metadata extraction
- [ ] Implement re-ranking for better relevance
- [ ] Add caching for frequent queries
- [ ] Support for document updates
- [ ] Add SQL database for metadata
- [ ] Implement access control and multi-tenancy

## References

- [LangChain Multimodal RAG](https://colab.research.google.com/gist/alejandro-ao/47db0b8b9d00b10a96ab42dd59d90b86/langchain-multimodal.ipynb)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
