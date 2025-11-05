import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_tags import st_tags

from src.models.model_manager import ModelManager
from src.models.sam_manager import SAMManager
from src.database.vector_db import VectorDatabase
from src.database.metadata_db import MetadataDatabase
from src.processing.segment_extractor import SegmentExtractor
from src.processing.video_processor import VideoProcessor

st.set_page_config(
    page_title="Booru Media Library",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

UPLOAD_DIR = "data/uploads"
VIDEO_DIR = "data/videos"
SEGMENTS_DIR = "data/segments"
FRAMES_DIR = "data/frames"

for directory in [UPLOAD_DIR, VIDEO_DIR, SEGMENTS_DIR, FRAMES_DIR]:
    os.makedirs(directory, exist_ok=True)

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

@st.cache_resource
def get_metadata_db():
    """Get metadata database instance"""
    return MetadataDatabase()

@st.cache_resource
def get_sam_manager():
    """Get SAM manager instance"""
    return SAMManager(model_type='vit_b')

@st.cache_resource
def get_video_processor():
    """Get video processor instance"""
    return VideoProcessor(output_dir=FRAMES_DIR)

def process_image_with_tags(image_file, model_manager, vector_db, metadata_db, tags=None, rating='safe'):
    """Process uploaded image with tags and metadata"""
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{image_file.name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(image_file.getbuffer())
    
    img = Image.open(filepath)
    width, height = img.size
    file_size = os.path.getsize(filepath)
    
    with st.spinner("Generating CLIP embedding..."):
        embedding = model_manager.encode(img)
        
        metadata = {
            'segment_id': file_id,
            'source': image_file.name,
            'type': 'full_image',
            'timestamp': datetime.now().isoformat(),
            'filepath': filepath,
            'rating': rating
        }
        
        vector_db.add_segments(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[file_id]
        )
        
        metadata_db.add_media(
            media_id=file_id,
            media_type='image',
            source_path=filepath,
            width=width,
            height=height,
            file_size=file_size,
            rating=rating
        )
        
        if tags:
            for tag in tags:
                metadata_db.add_media_tag(file_id, tag)
        
        return file_id, filepath

def process_image_with_segments(image_file, model_manager, vector_db, metadata_db, sam_manager,
                                  tags=None, rating='safe'):
    """Process uploaded image with segmentation"""
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{image_file.name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(image_file.getbuffer())
    
    img = Image.open(filepath)
    width, height = img.size
    file_size = os.path.getsize(filepath)
    
    metadata_db.add_media(
        media_id=file_id,
        media_type='image',
        source_path=filepath,
        width=width,
        height=height,
        file_size=file_size,
        rating=rating
    )
    
    if tags:
        for tag in tags:
            metadata_db.add_media_tag(file_id, tag)
    
    img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    with st.spinner("Generating segments..."):
        mask_generator = sam_manager.get_automatic_mask_generator()
        masks = mask_generator.generate(img_rgb)
        
        segment_extractor = SegmentExtractor(blur_kernel=21)
        
        segment_count = 0
        for idx, mask in enumerate(masks[:20]):
            segment_data = segment_extractor.extract_segment_with_blurred_background(mask, img_rgb)
            
            if segment_data is None:
                continue
            
            segment_id = f"{file_id}_seg_{idx:03d}"
            
            embedding = model_manager.encode(segment_data['pil_image'])
            
            segment_metadata = {
                'segment_id': segment_id,
                'source': f"{image_file.name} - Segment {idx}",
                'type': 'segment',
                'parent_media_id': file_id,
                'timestamp': datetime.now().isoformat(),
                'filepath': filepath,
                'bbox': segment_data['bbox'],
                'rating': rating
            }
            
            vector_db.add_segments(
                embeddings=[embedding],
                metadatas=[segment_metadata],
                ids=[segment_id]
            )
            
            metadata_db.add_segment(
                segment_id=segment_id,
                media_id=file_id,
                bbox=tuple(segment_data['bbox']),
                area=segment_data['area'],
                iou_score=segment_data['iou'],
                stability_score=segment_data['stability']
            )
            
            segment_count += 1
        
        return file_id, segment_count

def process_video_with_tags(video_file, model_manager, vector_db, metadata_db, video_processor,
                             extract_every_n=30, tags=None, rating='safe'):
    """Process uploaded video, extract frames, and generate embeddings"""
    video_id = str(uuid.uuid4())[:8]
    filename = f"{video_id}_{video_file.name}"
    filepath = os.path.join(VIDEO_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(video_file.getbuffer())
    
    file_size = os.path.getsize(filepath)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extracting video metadata...")
    video_metadata = video_processor.get_video_metadata(filepath)
    
    metadata_db.add_media(
        media_id=video_id,
        media_type='video',
        source_path=filepath,
        width=video_metadata['width'],
        height=video_metadata['height'],
        file_size=file_size,
        rating=rating
    )
    
    metadata_db.add_video(
        video_id=video_id,
        media_id=video_id,
        duration=video_metadata['duration'],
        fps=video_metadata['fps'],
        total_frames=video_metadata['total_frames'],
        codec=video_metadata['codec']
    )
    
    if tags:
        for tag in tags:
            metadata_db.add_media_tag(video_id, tag)
    
    status_text.text("Extracting frames from video...")
    
    def progress_callback(current, total):
        progress_bar.progress(min(current / total, 1.0))
    
    extracted_frames = video_processor.extract_frames(
        filepath,
        every_n_frames=extract_every_n,
        progress_callback=progress_callback
    )
    
    status_text.text(f"Processing {len(extracted_frames)} frames...")
    
    for idx, frame_info in enumerate(extracted_frames):
        frame_id = frame_info['frame_id']
        frame_path = frame_info['frame_path']
        
        metadata_db.add_frame(
            frame_id=frame_id,
            video_id=video_id,
            frame_number=frame_info['frame_number'],
            timestamp=frame_info['timestamp'],
            media_id=frame_id
        )
        
        img = Image.open(frame_path)
        embedding = model_manager.encode(img)
        
        frame_metadata = {
            'segment_id': frame_id,
            'source': f"{video_file.name} - Frame {frame_info['frame_number']}",
            'type': 'video_frame',
            'video_id': video_id,
            'frame_number': frame_info['frame_number'],
            'timestamp': frame_info['timestamp'],
            'filepath': frame_path,
            'rating': rating
        }
        
        vector_db.add_segments(
            embeddings=[embedding],
            metadatas=[frame_metadata],
            ids=[frame_id]
        )
        
        metadata_db.add_media(
            media_id=frame_id,
            media_type='frame',
            source_path=frame_path,
            width=video_metadata['width'],
            height=video_metadata['height'],
            rating=rating
        )
        
        progress_bar.progress((idx + 1) / len(extracted_frames))
    
    progress_bar.empty()
    status_text.empty()
    
    return video_id, len(extracted_frames)

def display_media_grid(results, metadata_db, vector_db, cols_per_row=4):
    """Display search results in booru-style grid with metadata"""
    if not results or 'ids' not in results or len(results['ids'][0]) == 0:
        st.info("No results found. Try different search criteria or upload some media!")
        return
    
    num_results = len(results['ids'][0])
    st.write(f"**Found {num_results} results**")
    
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
                media_id = results['ids'][0][idx]
                
                filepath = metadata.get('filepath', '')
                
                if os.path.exists(filepath):
                    st.image(filepath, use_container_width=True)
                    
                    media_tags = metadata_db.get_media_tags(media_id)
                    is_fav = metadata_db.is_favorite(media_id)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"**{metadata.get('source', 'Unknown')}**")
                    with col2:
                        fav_emoji = "⭐" if is_fav else "☆"
                        if st.button(fav_emoji, key=f"fav_{media_id}"):
                            if is_fav:
                                metadata_db.remove_favorite(media_id)
                            else:
                                metadata_db.add_favorite(media_id)
                            st.rerun()
                    
                    st.caption(f"Match: {similarity:.1%}")
                    
                    if media_tags:
                        tag_str = " ".join([f"[{t['tag_name']}]" for t in media_tags[:3]])
                        st.caption(tag_str)
                    
                    rating_badge = metadata.get('rating', 'safe')
                    rating_colors = {'safe': '🟢', 'questionable': '🟡', 'explicit': '🔴'}
                    st.caption(f"{rating_colors.get(rating_badge, '⚪')} {rating_badge}")
                else:
                    st.warning(f"Media not found")

def search_interface(model_manager, vector_db, metadata_db):
    """Advanced search interface with tag filtering"""
    st.header("🔍 Advanced Search")
    
    search_mode = st.radio("Search Mode", ["Text", "Image", "Tags"], horizontal=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if search_mode == "Text":
            query = st.text_input(
                "Search query",
                placeholder="e.g., 'red car', 'sunset landscape'"
            )
        elif search_mode == "Image":
            query_image = st.file_uploader("Upload reference image", type=['jpg', 'jpeg', 'png'])
        else:
            include_tags = st_tags(
                label="Include tags (AND)",
                text="Press enter to add",
                maxtags=10
            )
            exclude_tags = st_tags(
                label="Exclude tags (NOT)",
                text="Press enter to add",
                maxtags=5
            )
    
    with col2:
        n_results = st.number_input("Results", min_value=1, max_value=100, value=12)
        
        rating_filter = st.multiselect(
            "Rating",
            options=["safe", "questionable", "explicit"],
            default=["safe"]
        )
        
        sort_by = st.selectbox(
            "Sort by",
            ["Relevance", "Date (Newest)", "Score (Highest)", "Favorites"]
        )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if search_mode == "Text" and query:
            with st.spinner(f"Searching for '{query}'..."):
                query_embedding = model_manager.encode_text(query)
                results = vector_db.search_by_embedding(query_embedding, n_results=n_results)
                display_media_grid(results, metadata_db, vector_db)
        
        elif search_mode == "Image" and query_image:
            with st.spinner("Searching for similar images..."):
                img = Image.open(query_image)
                query_embedding = model_manager.encode(img)
                results = vector_db.search_by_embedding(query_embedding, n_results=n_results)
                display_media_grid(results, metadata_db, vector_db)
        
        elif search_mode == "Tags":
            with st.spinner("Searching by tags..."):
                media_ids = metadata_db.search_by_tags(
                    tags=include_tags,
                    exclude_tags=exclude_tags,
                    rating=rating_filter[0] if rating_filter else None,
                    limit=n_results
                )
                
                if media_ids:
                    results_list = []
                    for media_id in media_ids:
                        chroma_results = vector_db.collection.get(ids=[media_id])
                        if chroma_results and chroma_results['ids']:
                            results_list.append({
                                'id': media_id,
                                'metadata': chroma_results['metadatas'][0],
                                'distance': 0.0
                            })
                    
                    if results_list:
                        results = {
                            'ids': [[r['id'] for r in results_list]],
                            'metadatas': [[r['metadata'] for r in results_list]],
                            'distances': [[r['distance'] for r in results_list]]
                        }
                        display_media_grid(results, metadata_db, vector_db)
                    else:
                        st.info("No matching media found with those tags.")
                else:
                    st.info("No matching media found with those tags.")

def upload_interface(model_manager, vector_db, metadata_db, video_processor, sam_manager):
    """Upload interface for images and videos"""
    st.header("📤 Upload Media")
    
    media_type = st.radio("Upload Type", ["Images", "Video"], horizontal=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if media_type == "Images":
            uploaded_files = st.file_uploader(
                "Choose images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose video",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
    
    with col2:
        rating = st.selectbox("Rating", ["safe", "questionable", "explicit"])
        
        tags_input = st_tags(
            label="Add tags",
            text="Press enter to add",
            maxtags=20
        )
        
        if media_type == "Images":
            enable_segmentation = st.checkbox(
                "Enable segmentation",
                value=False,
                help="Extract individual objects using segmentation (slower but enables segment search)"
            )
        
        if media_type == "Video":
            extract_every = st.number_input(
                "Extract every N frames",
                min_value=1,
                max_value=120,
                value=30,
                help="Lower = more frames (slower)"
            )
    
    if media_type == "Images" and uploaded_files:
        if st.button("Process Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_count = 0
            segment_count = 0
            
            sam_manager.load_model()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                try:
                    if enable_segmentation:
                        media_id, segs = process_image_with_segments(
                            file, model_manager, vector_db, metadata_db, sam_manager,
                            tags=tags_input, rating=rating
                        )
                        segment_count += segs
                    else:
                        media_id, filepath = process_image_with_tags(
                            file, model_manager, vector_db, metadata_db,
                            tags=tags_input, rating=rating
                        )
                    processed_count += 1
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            progress_bar.empty()
            status_text.empty()
            
            if enable_segmentation:
                st.success(f"✓ Processed {processed_count} images with {segment_count} segments!")
            else:
                st.success(f"✓ Processed {processed_count} images successfully!")
            st.rerun()
    
    elif media_type == "Video" and uploaded_file:
        if st.button("Process Video", type="primary", use_container_width=True):
            try:
                video_id, frame_count = process_video_with_tags(
                    uploaded_file, model_manager, vector_db, metadata_db, video_processor,
                    extract_every_n=extract_every, tags=tags_input, rating=rating
                )
                st.success(f"✓ Processed video: {frame_count} frames extracted!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing video: {e}")

def favorites_interface(metadata_db, vector_db):
    """Display favorited media"""
    st.header("⭐ Favorites")
    
    favorite_ids = metadata_db.get_favorites()
    
    if not favorite_ids:
        st.info("No favorites yet! Click the ☆ icon on any media to add it to favorites.")
        return
    
    results_list = []
    for media_id in favorite_ids:
        chroma_results = vector_db.collection.get(ids=[media_id])
        if chroma_results and chroma_results['ids']:
            results_list.append({
                'id': media_id,
                'metadata': chroma_results['metadatas'][0],
                'distance': 0.0
            })
    
    if results_list:
        results = {
            'ids': [[r['id'] for r in results_list]],
            'metadatas': [[r['metadata'] for r in results_list]],
            'distances': [[r['distance'] for r in results_list]]
        }
        display_media_grid(results, metadata_db, vector_db)

def analytics_interface(metadata_db):
    """Analytics dashboard with statistics and visualizations"""
    st.header("📊 Analytics Dashboard")
    
    stats = metadata_db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Media", stats['total_media'])
    with col2:
        st.metric("Videos", stats['total_videos'])
    with col3:
        st.metric("Segments", stats['total_segments'])
    with col4:
        st.metric("Total Tags", stats['total_tags'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Popular Tags")
        popular_tags = metadata_db.get_popular_tags(limit=20)
        
        if popular_tags:
            tag_names = [t['tag_name'] for t in popular_tags]
            usage_counts = [t['usage_count'] for t in popular_tags]
            
            fig = px.bar(
                x=usage_counts,
                y=tag_names,
                orientation='h',
                labels={'x': 'Usage Count', 'y': 'Tag'},
                title="Top 20 Tags by Usage"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tags yet!")
    
    with col2:
        st.subheader("📈 Tag Cloud")
        
        if popular_tags:
            tag_cloud_data = {
                'tag': [t['tag_name'] for t in popular_tags],
                'count': [t['usage_count'] for t in popular_tags],
                'category': [t['category'] for t in popular_tags]
            }
            
            fig = px.scatter(
                tag_cloud_data,
                x=[i for i in range(len(tag_cloud_data['tag']))],
                y=[1] * len(tag_cloud_data['tag']),
                size='count',
                text='tag',
                color='category',
                title="Tag Cloud by Category"
            )
            fig.update_traces(textposition='middle center')
            fig.update_layout(
                showlegend=True,
                height=500,
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🎨 Booru Media Library")
    st.markdown("*Advanced media search with CLIP embeddings, tagging, and segmentation*")
    
    model_manager = load_models()
    vector_db = get_vector_db()
    metadata_db = get_metadata_db()
    sam_manager = get_sam_manager()
    video_processor = get_video_processor()
    
    if model_manager is None:
        st.error("Failed to load models. Please check the logs.")
        return
    
    with st.sidebar:
        st.header("📊 Library Stats")
        stats = metadata_db.get_statistics()
        st.metric("Total Media", stats['total_media'])
        st.metric("Favorites", stats['total_favorites'])
        st.metric("Total Tags", stats['total_tags'])
        
        st.divider()
        
        st.header("🔧 Settings")
        
        st.subheader("SAM Model")
        if sam_manager.is_downloaded():
            st.success("✓ SAM model downloaded")
        else:
            if st.button("📥 Download SAM Model", help="Download SAM for advanced segmentation"):
                with st.spinner("Downloading SAM model (~375 MB)..."):
                    try:
                        sam_manager.download_model()
                        st.success("✓ SAM model downloaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error downloading SAM: {e}")
        
        st.divider()
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.session_state.get('confirm_delete'):
                vector_db.delete_all()
                st.success("All data cleared!")
                st.session_state.confirm_delete = False
                st.rerun()
            else:
                st.session_state.confirm_delete = True
                st.warning("Click again to confirm")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload",
        "🔍 Search",
        "⭐ Favorites",
        "📊 Analytics",
        "ℹ️ About"
    ])
    
    with tab1:
        upload_interface(model_manager, vector_db, metadata_db, video_processor)
    
    with tab2:
        search_interface(model_manager, vector_db, metadata_db)
    
    with tab3:
        favorites_interface(metadata_db, vector_db)
    
    with tab4:
        analytics_interface(metadata_db)
    
    with tab5:
        st.header("About This Application")
        
        st.markdown("""
        ### 🎯 Features
        
        **Media Management**
        - Upload images and videos
        - Automatic frame extraction from videos
        - Tag-based organization with categories
        - Rating system (safe, questionable, explicit)
        - Favorites/bookmarking
        
        **Advanced Search**
        - Semantic text search using CLIP
        - Visual similarity search
        - Tag-based filtering with AND/NOT operators
        - Rating filters
        - Multiple sort options
        
        **Analytics**
        - Usage statistics
        - Popular tags visualization
        - Tag cloud
        - Library insights
        
        ### 🔬 Technology Stack
        
        - **CLIP (ViT-B/32)**: Vision-language embeddings
        - **ChromaDB**: Vector similarity search
        - **SQLite**: Metadata and tags
        - **SAM** (optional): Advanced segmentation
        - **Streamlit**: Modern web interface
        - **OpenCV & ImageIO**: Video processing
        
        ### 🚀 Future Enhancements
        
        - Full SAM integration for object segmentation
        - Segment-level search within images
        - Advanced filtering and sorting
        - Batch tagging operations
        - Export functionality
        - API access
        
        ### 💡 Booru-Style Features
        
        This application is inspired by booru imageboards like Gelbooru and Yande.re:
        - Tag-based organization
        - Advanced search operators
        - Rating system
        - Community-style features
        - Grid layout with metadata
        """)

if __name__ == "__main__":
    main()
