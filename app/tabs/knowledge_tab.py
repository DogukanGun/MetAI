"""Knowledge Base Management Tab"""

import streamlit as st
import tempfile
import os
from pathlib import Path


def render_knowledge_tab():
    """Render the Knowledge Base management tab."""
    
    st.header("Knowledge Base Management")
    st.write("Upload and manage documents for AI-powered meeting analysis")
    
    # Stats section
    st.subheader("Knowledge Base Statistics")
    
    try:
        from knowledge_base.ingestion.vector_store_manager import VectorStoreManager
        
        vector_store = VectorStoreManager(
            storage_path="./knowledge_base/storage",
            embedding_model="sentence-transformers"
        )
        
        stats = vector_store.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Text Chunks", stats.get('text_chunks', 0))
        with col2:
            st.metric("Images", stats.get('images', 0))
        with col3:
            st.metric("Text Store Size", stats.get('text_store_size', 0))
        with col4:
            st.metric("Image Store Size", stats.get('image_store_size', 0))
        
        if stats.get('text_chunks', 0) == 0:
            st.info("No documents uploaded yet. Upload your first document below!")
    
    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")
        st.info("Knowledge base will be initialized when you upload your first document.")
    
    st.markdown("---")
    
    # Upload section
    st.subheader("Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a document to add to the knowledge base",
        type=['pdf', 'txt', 'md', 'png', 'jpg', 'jpeg', 'docx'],
        help="Supported formats: PDF, TXT, MD, PNG, JPG, JPEG, DOCX"
    )
    
    if uploaded_file:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.write("**File Details:**")
        for key, value in file_details.items():
            st.text(f"{key}: {value}")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            extract_images = st.checkbox("Extract images from document", value=True)
            chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=1000)
        with col2:
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200)
        
        # Upload button
        if st.button("Upload to Knowledge Base", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    from knowledge_base.ingestion.document_processor import DocumentProcessor
                    from knowledge_base.ingestion.vector_store_manager import VectorStoreManager
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Process document
                    processor = DocumentProcessor(
                        storage_path="./knowledge_base/storage",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    metadata, text_chunks, images = processor.process_document(
                        file_path=temp_path,
                        extract_images=extract_images,
                        generate_summaries=False
                    )
                    
                    # Add to vector store
                    vector_store = VectorStoreManager(
                        storage_path="./knowledge_base/storage",
                        embedding_model="sentence-transformers"
                    )
                    vector_store.add_text_chunks(text_chunks)
                    vector_store.add_images(images)
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                    # Success message
                    st.success(f"Successfully processed {uploaded_file.name}!")
                    
                    # Show metadata
                    st.write("**Processing Results:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Text Chunks", metadata.num_text_chunks)
                    with col2:
                        st.metric("Images Extracted", metadata.num_images)
                    with col3:
                        st.metric("Pages", metadata.num_pages or "N/A")
                    
                    st.info("Document has been added to the knowledge base and is ready for AI analysis!")
                    
                    # Rerun to update stats
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Failed to process document: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Query test section
    st.subheader("Test Knowledge Base Query")
    st.write("Test retrieval from the knowledge base")
    
    query_text = st.text_input(
        "Enter a query",
        placeholder="e.g., What are the best practices for team meetings?"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    with col2:
        similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    if st.button("Query Knowledge Base") and query_text:
        with st.spinner("Searching knowledge base..."):
            try:
                from knowledge_base.retrieval.query_engine import QueryEngine
                from knowledge_base.ingestion.vector_store_manager import VectorStoreManager
                
                # Initialize components
                vector_store = VectorStoreManager(
                    storage_path="./knowledge_base/storage",
                    embedding_model="sentence-transformers"
                )
                query_engine = QueryEngine(vector_store)
                
                # Query
                text_chunks, images = query_engine.query(
                    query_text=query_text,
                    top_k=top_k,
                    include_images=False,
                    similarity_threshold=similarity_threshold
                )
                
                if text_chunks:
                    st.success(f"Found {len(text_chunks)} relevant chunks")
                    
                    for i, chunk in enumerate(text_chunks, 1):
                        with st.expander(f"Result {i} - Similarity: {chunk.similarity_score:.2%}"):
                            st.write(f"**Document ID**: {chunk.document_id}")
                            if chunk.page_number:
                                st.write(f"**Page**: {chunk.page_number}")
                            st.markdown("**Content:**")
                            st.write(chunk.metadata.get('content', 'Content not available'))
                else:
                    st.warning("No results found. Try lowering the similarity threshold or uploading more documents.")
            
            except Exception as e:
                st.error(f"Query failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Help section
    with st.expander("How to use the Knowledge Base"):
        st.markdown("""
        ### Upload Documents
        1. Select a document file (PDF, TXT, MD, or images)
        2. Configure processing options:
           - **Extract images**: Extract images from PDFs
           - **Chunk size**: Size of text chunks (default: 1000 characters)
           - **Chunk overlap**: Overlap between chunks (default: 200 characters)
        3. Click "Upload to Knowledge Base"
        
        ### What happens after upload?
        - Document is parsed and split into chunks
        - Text is embedded using Sentence Transformers
        - Images are embedded using CLIP
        - Content is stored in FAISS vector index
        - Ready for AI agent queries!
        
        ### Best Practices
        - Upload meeting guidelines and policies
        - Add team documentation
        - Include relevant research or standards
        - Update regularly with new information
        
        ### AI Agent Integration
        When analyzing meetings, the AI agent will:
        - Query the knowledge base based on emotions and content
        - Retrieve relevant context automatically
        - Cite sources in its analysis
        - Provide evidence-based recommendations
        
        ### Query Testing
        Use the "Test Knowledge Base Query" section to:
        - Verify documents are searchable
        - Check retrieval quality
        - Adjust similarity thresholds
        - Preview what the AI agent will see
        """)
    
    # API information
    with st.expander("API Access"):
        st.markdown("""
        ### Upload via API
        ```bash
        curl -X POST "http://localhost:8000/api/v1/knowledge/documents" \\
          -F "document=@path/to/file.pdf" \\
          -F "extract_images=true" \\
          -F "chunk_size=1000"
        ```
        
        ### Query via API
        ```bash
        curl -X POST "http://localhost:8000/api/v1/knowledge/query" \\
          -H "Content-Type: application/json" \\
          -d '{
            "query": "meeting best practices",
            "top_k": 5,
            "include_images": false,
            "similarity_threshold": 0.5
          }'
        ```
        
        ### Complete Analysis
        ```bash
        curl -X POST "http://localhost:8000/api/v1/analyze/complete" \\
          -F "video=@meeting.mp4" \\
          -F "enable_ai_agent=true"
        ```
        """)

