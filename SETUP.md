# Booru Media Library - Setup Guide

Complete installation and setup guide for the Booru Media Library application.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Model Downloads](#model-downloads)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or higher (3.10+ recommended)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5 GB free space (for models and data)
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support for faster processing)

### Recommended Requirements
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 6GB+ VRAM and CUDA 11.8+
- **Storage**: 10 GB+ SSD storage

---

## Installation

### Step 1: Clone or Download the Repository

```bash
git clone <your-repo-url>
cd booru-media-library
```

Or download and extract the ZIP file.

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**For GPU Support (NVIDIA CUDA):**

If you have an NVIDIA GPU with CUDA 11.8:
```bash
pip uninstall torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For other CUDA versions, visit: https://pytorch.org/get-started/locally/

### Step 4: Install System Dependencies

**FFmpeg** is required for video processing.

**On Windows:**
1. Download from: https://www.gyan.dev/ffmpeg/builds/
2. Extract and add to PATH
3. Verify: `ffmpeg -version`

**On macOS:**
```bash
brew install ffmpeg
```

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

---

## Model Downloads

The application uses two AI models:

### 1. CLIP Model (Required)

**Model**: OpenAI CLIP ViT-B/32  
**Size**: ~350 MB  
**Auto-download**: Yes

The CLIP model will automatically download on first run to:
- **Windows**: `C:\Users\<username>\.cache\clip\`
- **macOS/Linux**: `~/.cache/clip/`

No manual download needed!

### 2. SAM Model (Optional - for Advanced Segmentation)

**Model**: Segment Anything Model (SAM)  
**Size**: 375 MB (vit_b) | 1.25 GB (vit_l) | 2.56 GB (vit_h)  
**Auto-download**: Via app UI or manual

#### Option A: Download via Application UI (Recommended)

1. Start the application
2. Go to the sidebar → **Settings** section
3. Click **"📥 Download SAM Model"**
4. Wait for download to complete (~375 MB for vit_b)

#### Option B: Manual Download

Download the model file:

**SAM ViT-B (Recommended - 375 MB):**
```bash
mkdir -p models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
```

**SAM ViT-L (Larger, more accurate - 1.25 GB):**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O models/sam_vit_l_0b3195.pth
```

**SAM ViT-H (Huge, best quality - 2.56 GB):**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth
```

**For Windows users** (using PowerShell):
```powershell
mkdir models -Force
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "models\sam_vit_b_01ec64.pth"
```

**Note**: If you don't download SAM, the app will use fallback segmentation methods (OpenCV SLIC/Selective Search).

---

## Configuration

### Directory Structure

The application will automatically create these directories on first run:

```
booru-media-library/
├── data/
│   ├── uploads/        # Uploaded images
│   ├── videos/         # Uploaded videos
│   ├── segments/       # Extracted segments
│   ├── frames/         # Video frames
│   └── metadata.db     # SQLite database
├── chroma_db/          # Vector database
├── models/             # SAM models (if downloaded)
├── .streamlit/
│   ├── config.toml     # Streamlit config
│   └── style.css       # Custom CSS
└── src/                # Source code
```

### Streamlit Configuration

The `.streamlit/config.toml` file contains default settings. You can customize:

```toml
[server]
port = 5000
address = "0.0.0.0"

[theme]
primaryColor = "#0075f8"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f5f5"
textColor = "#333333"
```

---

## Running the Application

### Start the Application

**Standard method:**
```bash
streamlit run app.py --server.port 5000
```

**Access the application:**
- Open your browser to: http://localhost:5000

### Running on a Different Port

```bash
streamlit run app.py --server.port 8501
```

### Running for Network Access

To access from other devices on your network:
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 5000
```

Then access via: `http://<your-local-ip>:5000`

---

## Usage Guide

### 1. Upload Media

**Upload Tab** → Choose upload type:
- **Images**: Supports JPG, JPEG, PNG (max 200MB)
- **Videos**: Supports MP4, AVI, MOV

**Options:**
- Add tags (comma-separated)
- Set rating (safe/questionable/explicit)
- Enable segmentation (extracts objects)
- Set frame interval for videos

### 2. Search Media

**Search Tab** → Choose search method:
- **Text Search**: Enter description (e.g., "sunset over ocean")
- **Tag Search**: Search by tags
- **Image Search**: Upload reference image
- **Segment Search**: Find similar segments

### 3. Browse Segments

**Segments Tab** → View all extracted objects:
- Filter by source type (image/video/frame)
- Search segments by text
- Find similar segments

### 4. Manage Favorites

**Favorites Tab** → Bookmark your favorite media
- View all favorited items
- Quick access to bookmarked content

### 5. View Analytics

**Analytics Tab** → Library statistics:
- Tag usage distribution
- Popular tags
- Media counts by type
- Rating distribution

---

## Troubleshooting

### Issue: Models not downloading

**Solution:**
- Check internet connection
- Verify firewall settings
- Try manual download (see Model Downloads section)
- Check disk space availability

### Issue: Out of memory errors

**Solution:**
- Close other applications
- Reduce number of segments extracted (use higher confidence threshold)
- Process videos at larger frame intervals
- Use CPU instead of GPU if GPU memory is low

### Issue: Video processing fails

**Solution:**
- Verify FFmpeg is installed: `ffmpeg -version`
- Check video codec compatibility
- Try converting video to MP4 with H.264 codec
- Reduce video resolution or length

### Issue: Slow performance

**Optimization tips:**
- **Use GPU**: Install CUDA-compatible PyTorch
- **Reduce segments**: Disable segmentation for faster uploads
- **Limit search results**: Search with more specific queries
- **Use vit_b SAM**: Smaller/faster than vit_l or vit_h

### Issue: Application won't start

**Solution:**
1. Check Python version: `python --version` (should be 3.9+)
2. Verify all dependencies: `pip install -r requirements.txt`
3. Check for port conflicts (change port if 5000 is in use)
4. Review terminal error messages
5. Delete `chroma_db/` and restart (will lose vector data)

### Issue: Streamlit errors about missing modules

**Solution:**
```bash
pip install --upgrade streamlit
pip install --upgrade -r requirements.txt
```

---

## Performance Tips

### For Faster Processing

1. **Use GPU acceleration**
   - Install CUDA toolkit
   - Install CUDA-compatible PyTorch

2. **Optimize video settings**
   - Use larger frame intervals (e.g., 30 instead of 10)
   - Process shorter video clips
   - Lower video resolution before upload

3. **Manage database size**
   - Periodically clear unused media
   - Remove duplicate segments
   - Archive old data

### For Better Results

1. **Image uploads**
   - Use high-quality images (good lighting, clear subjects)
   - Add descriptive tags
   - Enable segmentation for object-level search

2. **Search queries**
   - Be specific in text descriptions
   - Combine multiple tags for precision
   - Use exclusion tags to filter out unwanted results

---

## Advanced Configuration

### Using Different SAM Models

To use vit_l or vit_h instead of vit_b:

Edit `app.py`:
```python
@st.cache_resource
def get_sam_manager():
    return SAMManager(model_type='vit_l')  # or 'vit_h'
```

### Custom Theme

Edit `.streamlit/style.css` to customize colors, fonts, and layout.

### Database Backup

**Backup your data:**
```bash
# Backup metadata
cp data/metadata.db data/metadata.db.backup

# Backup vector database
cp -r chroma_db chroma_db.backup
```

---

## Support

For issues, bugs, or feature requests:
- Check the troubleshooting section above
- Review error messages in the terminal
- Check browser console for frontend errors (F12)

---

## License & Credits

**CLIP**: OpenAI (https://github.com/openai/CLIP)  
**SAM**: Meta Research (https://github.com/facebookresearch/segment-anything)  
**ChromaDB**: Chroma (https://www.trychroma.com/)  
**Streamlit**: Streamlit Inc. (https://streamlit.io/)

---

**Enjoy using Booru Media Library! 🎨**
