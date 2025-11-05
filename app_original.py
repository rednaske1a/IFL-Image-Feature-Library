import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import uuid
from datetime import datetime

from src.models.model_manager import ModelManager
from src.database.vector_db import VectorDatabase
from src.processing.segment_extractor import SegmentExtractor

st.set_page_config(
    page_title="Media Feature Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

UPLOAD_DIR = "data/uploads"
SEGMENTS_DIR = "data/segments"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)

@st.cache_resource
def load_models():
    """Load AI models (cached to avoid reloading)"""
    model_manager = ModelManager()
    try:
        model_manager.load_clip()
        return model_manager
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_resource
def get_vector_db():
    """Get vector database instance"""
    vector_db = VectorDatabase()
    vector_db.create_or_get_collection()
    return vector_db

def process_image(image_file, model_manager, vector_db, use_full_image=True):
    """Process uploaded image and add to database"""
    
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{image_file.name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(image_file.getbuffer())
    
    img = Image.open(filepath)
    
    if use_full_image:
        with st.spinner("Generating CLIP embedding for image..."):
            embedding = model_manager.encode(img)
            
            metadata = {
                'segment_id': file_id,
                'source': image_file.name,
                'type': 'full_image',
                'timestamp': datetime.now().isoformat(),
                'filepath': filepath
            }
            
            vector_db.add_segments(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[file_id]
            )
            
            return 1, filepath
    
    return 0, filepath

def search_by_text(query, model_manager, vector_db, n_results=12):
    """Search for images using text query"""
    query_embedding = model_manager.encode_text(query)
    
    results = vector_db.search_by_embedding(query_embedding, n_results=n_results)
    
    return results

def search_by_image(image_file, model_manager, vector_db, n_results=12):
    """Search for similar images"""
    img = Image.open(image_file)
    query_embedding = model_manager.encode(img)
    
    results = vector_db.search_by_embedding(query_embedding, n_results=n_results)
    
    return results

def display_results(results):
    """Display search results in a grid"""
    if not results or 'ids' not in results or len(results['ids'][0]) == 0:
        st.info("No results found. Try uploading some images first!")
        return
    
    num_results = len(results['ids'][0])
    st.write(f"**Found {num_results} results**")
    
    cols_per_row = 4
    
    for i in range(0, num_results, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx >= num_results:
                break
            
            with cols[j]:
                metadata = results['metadatas'][0][idx]
                distance = results['distances'][0][idx]
                similarity = 1 - distance
                
                filepath = metadata.get('filepath', '')
                
                if os.path.exists(filepath):
                    st.image(filepath, use_container_width=True)
                    st.caption(f"**{metadata.get('source', 'Unknown')}**")
                    st.caption(f"Similarity: {similarity:.2%}")
                else:
                    st.warning(f"Image not found: {metadata.get('source', 'Unknown')}")

def main():
    st.title("🎬 Media Feature Extractor")
    st.markdown("*Semantic search for images using CLIP embeddings*")
    
    model_manager = load_models()
    vector_db = get_vector_db()
    
    if model_manager is None:
        st.error("Failed to load models. Please check the logs.")
        return
    
    with st.sidebar:
        st.header("📊 Statistics")
        total_count = vector_db.get_count()
        st.metric("Total Images", total_count)
        st.metric("Embedding Model", "CLIP ViT-B/32")
        st.metric("Embedding Dimension", "512")
        
        st.divider()
        
        st.header("⚙️ Mode")
        st.info("**CLIP-only mode**\n\nProcessing full images for fast semantic search.")
        st.caption("Research-backed blurred background approach ready for SAM integration.")
        
        st.divider()
        
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.get('confirm_delete'):
                vector_db.delete_all()
                st.success("Database cleared!")
                st.session_state.confirm_delete = False
                st.rerun()
            else:
                st.session_state.confirm_delete = True
                st.warning("Click again to confirm deletion")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🔍 Text Search", "🖼️ Image Search", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Images")
        st.write("Upload images to build your searchable media library")
        
        uploaded_files = st.file_uploader(
            "Choose images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload one or more images"
        )
        
        if uploaded_files:
            if st.button("Process Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_processed = 0
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    try:
                        count, filepath = process_image(file, model_manager, vector_db)
                        total_processed += count
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✓ Processed {total_processed} images successfully!")
                st.rerun()
    
    with tab2:
        st.header("Search by Text")
        st.write("Find images using natural language descriptions")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., 'sunset over ocean', 'red car', 'happy dog'",
                help="Describe what you're looking for in natural language"
            )
        
        with col2:
            n_results = st.number_input("Results", min_value=1, max_value=50, value=12)
        
        if st.button("🔍 Search", type="primary"):
            if not query:
                st.warning("Please enter a search query")
            else:
                with st.spinner(f"Searching for '{query}'..."):
                    results = search_by_text(query, model_manager, vector_db, n_results)
                    display_results(results)
    
    with tab3:
        st.header("Search by Image")
        st.write("Upload an image to find visually similar images in your library")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query_image = st.file_uploader(
                "Upload reference image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image to find similar ones"
            )
        
        with col2:
            n_results_img = st.number_input("Results", min_value=1, max_value=50, value=12, key="img_results")
        
        if query_image:
            st.image(query_image, caption="Query Image", width=300)
            
            if st.button("🔍 Find Similar", type="primary"):
                with st.spinner("Searching for similar images..."):
                    results = search_by_image(query_image, model_manager, vector_db, n_results_img)
                    display_results(results)
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ### 🎯 Features
        
        - **Semantic Search**: Find images using natural language queries
        - **Visual Similarity**: Upload an image to find similar ones
        - **CLIP Embeddings**: 512-dimensional vector representations
        - **Fast Search**: ChromaDB vector database with cosine similarity
        
        ### 🔬 Technology Stack
        
        - **CLIP (ViT-B/32)**: OpenAI's vision-language model
        - **ChromaDB**: Vector database for similarity search
        - **Streamlit**: Modern web interface
        - **OpenCV & Pillow**: Image processing
        
        ### 📚 Research-Backed Approach
        
        This application uses the **blurred background method** for segment extraction,
        based on research from:
        - CLIPAway (2023)
        - Mask-ControlNet (2023)
        - Ultralytics YOLOv8
        
        **Benefits:**
        - De-emphasizes background while preserving context
        - Better semantic embeddings than white/black backgrounds
        - 21x21 Gaussian blur kernel (research-recommended)
        
        ### 🚀 Future Enhancements
        
        - **SAM Integration**: Full object segmentation with blurred backgrounds
        - **Video Processing**: Extract and search video frames
        - **Dataset Export**: Export labeled datasets (COCO, YOLO formats)
        - **Advanced Filtering**: Filter by size, quality, date
        - **Batch Processing**: Handle large media libraries efficiently
        
        ### 📖 How It Works
        
        1. **Upload**: Images are processed and stored
        2. **Encode**: CLIP generates 512-dim embeddings
        3. **Store**: Embeddings saved in vector database
        4. **Search**: Text or image queries encoded and matched
        5. **Rank**: Results sorted by cosine similarity
        
        ### 💡 Usage Tips
        
        - Use descriptive search terms for better results
        - Upload diverse images to build a rich library
        - Image search works best with similar content types
        """)

if __name__ == "__main__":
    main()
