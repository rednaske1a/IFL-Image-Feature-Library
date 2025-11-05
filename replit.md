# Booru Media Library - Advanced Semantic Search Application

## Overview

A comprehensive booru-style media library application featuring CLIP embeddings for semantic search, SAM segmentation for object extraction, tag-based organization, and advanced search capabilities. Built with Streamlit and inspired by popular booru sites like gelbooru.com and yande.re.

The application enables semantic search where users can find images using natural language queries like "sunset over ocean" or "red car" without manual tagging. It supports object segmentation to extract and search individual elements from images and videos, advanced tag autocomplete with category support, favorites/bookmarks, rating systems, and comprehensive analytics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with booru-style multi-tab interface
- **Tabs**: Upload, Search, Segments, Favorites, Analytics, About
- **State Management**: Streamlit's built-in caching (`@st.cache_resource`) for model and database instances
- **UI Components**: File uploaders, tag autocomplete with streamlit-tags, search inputs, grid galleries, segment browsers, progress indicators
- **Grid View**: Responsive media grid similar to gelbooru/yande.re with thumbnails and metadata overlay

### Backend Architecture

**Core Processing Pipeline**:
1. Image upload → PIL Image loading
2. CLIP preprocessing → 512-dimensional embedding generation
3. Vector storage in ChromaDB with metadata
4. Similarity search using cosine distance

**Key Components**:

- **ModelManager** (`src/models/model_manager.py`): Manages CLIP model lifecycle
  - Lazy loads CLIP ViT-B/32 model on first use
  - Handles device selection (CUDA vs CPU)
  - Provides embedding encoding interface
  - Caches model in memory to avoid reloading

- **VectorDatabase** (`src/database/vector_db.py`): ChromaDB wrapper for vector operations
  - Creates persistent collection for media segments
  - Stores embeddings with associated metadata
  - Executes similarity queries with configurable result limits
  - Uses cosine similarity for matching

- **MetadataDatabase** (`src/database/metadata_db.py`): SQLite database for structured metadata
  - Stores media information (dimensions, file size, upload date)
  - Supports ratings, scores, view counts, favorites
  - Video-specific metadata (duration, FPS)
  - Tag system for manual categorization

- **Vendored CLIP** (`src/vendored_clip/`): Custom CLIP implementation
  - Self-contained to avoid torchvision dependency
  - Uses PIL and NumPy for image preprocessing
  - Downloads model weights automatically (~350MB for ViT-B/32)
  - Supports multiple model variants (RN50, ViT-B/32, ViT-L/14)

**Design Decisions**:

- **Vendored CLIP over pip package**: Reduces dependency conflicts and ensures consistent preprocessing without torchvision
- **ChromaDB for vectors**: Provides built-in persistence, no separate database server needed, good for desktop/single-user applications
- **SQLite for metadata**: Lightweight, serverless, sufficient for local application needs
- **Streamlit over CustomTkinter**: Faster development, easier deployment, web-accessible, though original codebase shows desktop GUI was considered

**Advanced Features** (fully implemented):
- **SegmentExtractor**: Object segmentation with blurred background approach for better embeddings (based on CLIPAway, Mask-ControlNet research)
  - Saves cropped segments to disk in `data/segments/` for browsing
  - Creates CLIP embeddings from extracted segments for semantic search
  - Stores segment metadata including bounding boxes and parent media references
- **SAMManager**: Integration with Segment Anything Model for automatic object detection (fallback to OpenCV SLIC/Selective Search when SAM weights unavailable)
- **VideoProcessor**: Frame extraction from videos with metadata tracking and segmentation support
  - Extracts keyframes from videos
  - Applies SAM segmentation to each frame
  - Creates searchable segment library from video content
- **Segment Library**: Dedicated interface for browsing all extracted segments
  - Text search across segments using CLIP embeddings
  - Image-based similarity search
  - Filter by source type (image/video/frame)
  - Display segment thumbnails with metadata
- **Tag System**: Booru-style tag autocomplete and organization
  - Tag categories: general, character, artist, series, meta
  - Popular tags display with usage counts
  - Advanced search with tag combination and exclusion
- **Favorites System**: Bookmark media and segments for quick access
- **Analytics Dashboard**: Statistics on library size, tag usage, rating distribution

### Data Storage

**Vector Database**:
- **Technology**: ChromaDB with persistent client
- **Location**: `./chroma_db` directory
- **Collection**: Single collection named "media_segments"
- **Embedding Dimension**: 512 (CLIP ViT-B/32 output)
- **Similarity Metric**: Cosine similarity

**Metadata Database**:
- **Technology**: SQLite3
- **Location**: `./data/metadata.db`
- **Tables**: 
  - `media` - core media information
  - `videos` - video-specific metadata
  - `tags` - tag associations
  - `segments` - object segment tracking
- **Relationships**: Media ID as foreign key for related data

**File Storage**:
- `data/uploads/` - Original uploaded images
- `data/videos/` - Video files
- `data/segments/` - Extracted object segments
- `data/frames/` - Video frame extractions

### Authentication & Authorization
No authentication system implemented. Application designed for single-user local deployment.

## External Dependencies

### AI Models
- **CLIP ViT-B/32**: OpenAI's vision-language model
  - Source: OpenAI public CDN (openaipublic.azureedge.net)
  - Size: ~350MB
  - Auto-downloads to `~/.cache/clip/`
  - Used for: Image and text embedding generation

- **SAM (Segment Anything Model)**: Optional object segmentation
  - Source: Facebook Research (dl.fbaipublicfiles.com)
  - Variants: vit_h (2.56GB), vit_l (1.25GB), vit_b (375MB)
  - Storage: `./models/` directory
  - Status: Optional, falls back to OpenCV when unavailable

### Python Libraries
- **torch**: PyTorch deep learning framework for model inference
- **chromadb**: Vector database for similarity search
- **streamlit**: Web application framework
- **Pillow (PIL)**: Image loading and manipulation
- **opencv-python (cv2)**: Video processing, image operations
- **numpy**: Numerical operations on arrays
- **ftfy**: Text normalization for CLIP tokenizer
- **regex**: Pattern matching for CLIP tokenizer
- **tqdm**: Progress bars for downloads
- **imageio**: Video reading alternative

### Optional Dependencies
- **streamlit-tags**: Tag input component (imported but may not be actively used)
- **plotly**: Visualization library (imported but may not be actively used in current version)

### System Dependencies
- **ffmpeg**: Required for video processing (via opencv-python)
- **CUDA** (optional): GPU acceleration for faster embedding generation