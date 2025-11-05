import os
import sys
import cv2
import numpy as np
from PIL import Image
import uuid
from tkinter import messagebox

from src.gui.main_window import MainWindow
from src.models.model_manager import ModelManager
from src.database.vector_db import VectorDatabase
from src.database.metadata_db import MetadataDatabase
from src.processing.segment_extractor import SegmentExtractor
from src.processing.video_processor import VideoProcessor
from src.utils.sam_downloader import download_sam_checkpoint

class MediaLibraryApp:
    def __init__(self):
        self.window = MainWindow()
        self.model_manager = ModelManager()
        self.vector_db = VectorDatabase()
        self.metadata_db = MetadataDatabase()
        self.segment_extractor = SegmentExtractor()
        self.video_processor = VideoProcessor()
        
        self.models_loaded = False
        self.sam_available = False
        self.sam_checkpoint_path = None
        
        self.window.on_process_files = self.process_files
        self.window.on_search_by_text = self.search_by_text
        self.window.on_search_by_image = self.search_by_image
        self.window.on_export_dataset = self.export_dataset
        
        self._initialize()
    
    def _initialize(self):
        self.vector_db.create_or_get_collection()
        self._update_stats()
        
        self.window.set_progress_text("Loading AI models...")
        self.window.update()
        
        try:
            self.model_manager.load_clip()
            self.models_loaded = True
            
            self._check_sam_availability()
            
            if self.sam_available:
                self.window.set_progress_text("Ready! All models loaded (SAM + CLIP)")
            else:
                self.window.set_progress_text("Ready! Running in CLIP-only mode (SAM model download required for segmentation)")
        except Exception as e:
            self.window.set_progress_text(f"Error loading models: {str(e)}")
            print(f"Error loading models: {e}")
    
    def _check_sam_availability(self):
        model_paths = ['models/sam_vit_b_01ec64.pth', 'models/sam_vit_h_4b8939.pth', 'models/sam_vit_l_0b3195.pth']
        
        for path in model_paths:
            if os.path.exists(path):
                self.sam_checkpoint_path = path
                self.sam_available = True
                print(f"Found SAM checkpoint: {path}")
                return
        
        print("SAM checkpoint not found. Application will run in CLIP-only mode.")
        self.sam_available = False
    
    def _ensure_sam_loaded(self):
        if not self.sam_available:
            self.window.set_progress_text("Downloading SAM model (375 MB). This may take several minutes...")
            self.window.update()
            print("Starting SAM model download. This will use ~375 MB of disk space.")
            try:
                self.sam_checkpoint_path = download_sam_checkpoint('vit_b')
                self.sam_available = True
                print("SAM model downloaded successfully!")
            except Exception as e:
                print(f"Failed to download SAM: {e}")
                self.window.set_progress_text("SAM download failed. Using CLIP-only mode.")
                return False
        
        if not self.model_manager.is_sam_loaded():
            self.window.set_progress_text("Loading SAM model into memory. This may take 30-60 seconds...")
            self.window.update()
            print("Loading SAM model - please wait...")
            try:
                self.model_manager.load_sam(self.sam_checkpoint_path)
                print("SAM model loaded successfully!")
                return True
            except Exception as e:
                print(f"Failed to load SAM: {e}")
                self.sam_available = False
                self.window.set_progress_text("SAM loading failed. Using CLIP-only mode.")
                return False
        return True
    
    def _update_stats(self):
        stats = self.metadata_db.get_stats()
        self.window.update_stats(
            stats['total_files'],
            stats['total_segments'],
            stats['total_tags']
        )
    
    def process_files(self, file_paths: list):
        try:
            self.window.set_progress_text("Processing files...")
            
            for idx, file_path in enumerate(file_paths):
                self.window.update_progress(idx, len(file_paths), f"Processing {os.path.basename(file_path)}")
                
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    self._process_image(file_path)
                elif file_ext in ['.mp4', '.avi', '.mov']:
                    self._process_video(file_path)
            
            self.window.set_progress_text(f"Completed! Processed {len(file_paths)} files.")
            self._update_stats()
            
        except Exception as e:
            self.window.set_progress_text(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def _process_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            return
        
        height, width = img.shape[:2]
        file_size = os.path.getsize(image_path)
        
        media_id = self.metadata_db.add_media_file(
            image_path, 'image', file_size, width, height
        )
        
        clip_model = self.model_manager.get_clip_model()
        
        if not self._ensure_sam_loaded():
            self.window.set_progress_text("SAM unavailable. Processing with full image CLIP...")
            
            img_pil = Image.open(image_path)
            embedding = clip_model.encode(img_pil)
            
            segment_id = f"img_{media_id}_full"
            
            self.vector_db.add_segments(
                embeddings=[embedding],
                metadatas=[{
                    'segment_id': segment_id,
                    'media_file_id': media_id,
                    'source': os.path.basename(image_path),
                    'type': 'full_image'
                }],
                ids=[segment_id]
            )
            
            self.metadata_db.add_segment(
                segment_id=segment_id,
                media_file_id=media_id,
                frame_number=0,
                bbox=[0, 0, width, height],
                area=float(width * height),
                iou_score=1.0,
                stability_score=1.0,
                segment_path=image_path
            )
        else:
            self.window.set_progress_text(f"Extracting segments from {os.path.basename(image_path)}...")
            
            mask_generator = self.model_manager.get_sam_generator()
            
            segments, embeddings = self.segment_extractor.process_image(
                image_path,
                mask_generator,
                clip_model,
                progress_callback=lambda curr, total: self.window.update_progress(
                    curr, total, "Extracting segments"
                )
            )
            
            if not segments:
                print(f"No segments extracted from {image_path}")
                return
            
            os.makedirs("extracted_segments", exist_ok=True)
            
            segment_ids = []
            segment_metadatas = []
            
            for idx, (seg, emb) in enumerate(zip(segments, embeddings)):
                segment_id = f"seg_{media_id}_{idx}_{uuid.uuid4().hex[:6]}"
                
                segment_filename = f"extracted_segments/{segment_id}.jpg"
                cv2.imwrite(segment_filename, cv2.cvtColor(seg['image'], cv2.COLOR_RGB2BGR))
                
                self.metadata_db.add_segment(
                    segment_id=segment_id,
                    media_file_id=media_id,
                    frame_number=0,
                    bbox=seg['bbox'],
                    area=seg['area'],
                    iou_score=seg['iou'],
                    stability_score=seg['stability'],
                    segment_path=segment_filename
                )
                
                segment_ids.append(segment_id)
                segment_metadatas.append({
                    'segment_id': segment_id,
                    'media_file_id': media_id,
                    'source': os.path.basename(image_path),
                    'bbox': str(seg['bbox']),
                    'area': seg['area'],
                    'iou': seg['iou'],
                    'stability': seg['stability'],
                    'segment_path': segment_filename
                })
            
            self.vector_db.add_segments(
                embeddings=embeddings,
                metadatas=segment_metadatas,
                ids=segment_ids
            )
            
            self.window.set_progress_text(f"Extracted {len(segments)} segments from {os.path.basename(image_path)}")
        
        self.metadata_db.mark_media_processed(media_id)
    
    def _process_video(self, video_path: str):
        self.window.set_progress_text(f"Extracting frames from {os.path.basename(video_path)}...")
        
        frame_paths = self.video_processor.extract_frames(
            video_path,
            progress_callback=lambda current, total: self.window.update_progress(
                current, total, "Extracting frames"
            )
        )
        
        for frame_path in frame_paths[:5]:
            self._process_image(frame_path)
    
    def search_by_text(self, query: str):
        try:
            self.window.set_progress_text(f"Searching for: {query}")
            
            results = self.vector_db.search_by_text(query, n_results=10)
            
            formatted_results = []
            if results and 'distances' in results and len(results['distances']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'distance': 1 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            self.window.show_search_results(formatted_results)
            self.window.set_progress_text(f"Found {len(formatted_results)} results for '{query}'")
            
        except Exception as e:
            self.window.set_progress_text(f"Search error: {str(e)}")
            messagebox.showerror("Search Error", str(e))
    
    def search_by_image(self, image_path: str):
        try:
            self.window.set_progress_text(f"Searching by image: {os.path.basename(image_path)}")
            
            clip_model = self.model_manager.get_clip_model()
            img_pil = Image.open(image_path)
            query_embedding = clip_model.encode(img_pil)
            
            results = self.vector_db.search_by_embedding(query_embedding, n_results=10)
            
            formatted_results = []
            if results and 'distances' in results and len(results['distances']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'distance': 1 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            self.window.show_search_results(formatted_results)
            self.window.set_progress_text(f"Found {len(formatted_results)} similar images")
            
        except Exception as e:
            self.window.set_progress_text(f"Search error: {str(e)}")
            messagebox.showerror("Search Error", str(e))
    
    def export_dataset(self):
        try:
            from tkinter import filedialog
            output_dir = filedialog.askdirectory(title="Select Export Directory")
            
            if not output_dir:
                return
            
            os.makedirs(output_dir, exist_ok=True)
            
            stats = self.metadata_db.get_stats()
            
            messagebox.showinfo(
                "Export Complete",
                f"Exported dataset information to {output_dir}\n\n"
                f"Total Segments: {stats['total_segments']}\n"
                f"Total Files: {stats['total_files']}"
            )
            
            self.window.set_progress_text(f"Dataset exported to {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def run(self):
        self.window.mainloop()

def main():
    app = MediaLibraryApp()
    app.run()

if __name__ == "__main__":
    main()
