import cv2
import os
import uuid
from typing import List, Tuple, Optional, Callable
import imageio.v3 as iio
from datetime import datetime
from PIL import Image

class VideoProcessor:
    """
    Video processing module for extracting frames and metadata
    """
    
    def __init__(self, output_dir: str = "./data/frames"):
        """
        Initialize video processor
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def get_video_metadata(self, video_path: str) -> dict:
        """
        Extract video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict with fps, duration, total_frames, width, height, codec
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        duration = total_frames / fps if fps > 0 else 0
        
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration,
            'codec': codec_str
        }
    
    def extract_frames(self, 
                       video_path: str,
                       every_n_frames: int = 30,
                       max_frames: Optional[int] = None,
                       progress_callback: Optional[Callable] = None) -> List[dict]:
        """
        Extract frames from video at regular intervals
        
        Args:
            video_path: Path to video file
            every_n_frames: Extract every Nth frame (default: 30 = 1 fps for 30fps video)
            max_frames: Maximum number of frames to extract (None = no limit)
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of dicts with frame_path, frame_number, timestamp
        """
        video_id = str(uuid.uuid4())[:8]
        metadata = self.get_video_metadata(video_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = metadata['fps']
        total_frames = metadata['total_frames']
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        video_frame_dir = os.path.join(self.output_dir, f"video_{video_id}")
        os.makedirs(video_frame_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % every_n_frames == 0:
                if max_frames and saved_count >= max_frames:
                    break
                
                frame_id = f"{video_id}_frame_{saved_count:06d}"
                timestamp = frame_count / fps if fps > 0 else 0
                frame_path = os.path.join(video_frame_dir, f"{frame_id}.jpg")
                
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                extracted_frames.append({
                    'frame_id': frame_id,
                    'frame_path': frame_path,
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'video_id': video_id
                })
                
                saved_count += 1
                
                if progress_callback:
                    progress_callback(frame_count, total_frames)
            
            frame_count += 1
        
        cap.release()
        
        return extracted_frames
    
    def extract_frames_imageio(self,
                                video_path: str,
                                every_n_frames: int = 30,
                                max_frames: Optional[int] = None,
                                progress_callback: Optional[Callable] = None) -> List[dict]:
        """
        Extract frames using ImageIO (faster for bulk extraction)
        
        Args:
            video_path: Path to video file
            every_n_frames: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of dicts with frame_path, frame_number, timestamp
        """
        video_id = str(uuid.uuid4())[:8]
        metadata = self.get_video_metadata(video_path)
        fps = metadata['fps']
        
        video_frame_dir = os.path.join(self.output_dir, f"video_{video_id}")
        os.makedirs(video_frame_dir, exist_ok=True)
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        for idx, frame in enumerate(iio.imiter(video_path)):
            if idx % every_n_frames == 0:
                if max_frames and saved_count >= max_frames:
                    break
                
                frame_id = f"{video_id}_frame_{saved_count:06d}"
                timestamp = idx / fps if fps > 0 else 0
                frame_path = os.path.join(video_frame_dir, f"{frame_id}.jpg")
                
                pil_image = Image.fromarray(frame)
                pil_image.save(frame_path, quality=95)
                
                extracted_frames.append({
                    'frame_id': frame_id,
                    'frame_path': frame_path,
                    'frame_number': idx,
                    'timestamp': timestamp,
                    'video_id': video_id
                })
                
                saved_count += 1
                
                if progress_callback:
                    progress_callback(idx, metadata['total_frames'])
            
            frame_count = idx
        
        return extracted_frames
    
    def extract_evenly_spaced_frames(self,
                                      video_path: str,
                                      n_frames: int = 10,
                                      progress_callback: Optional[Callable] = None) -> List[dict]:
        """
        Extract N evenly spaced frames from video
        
        Args:
            video_path: Path to video file
            n_frames: Number of frames to extract
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of dicts with frame_path, frame_number, timestamp
        """
        metadata = self.get_video_metadata(video_path)
        total_frames = metadata['total_frames']
        
        if total_frames <= n_frames:
            frame_indices = list(range(total_frames))
        else:
            interval = total_frames / n_frames
            frame_indices = [int(i * interval) for i in range(n_frames)]
        
        video_id = str(uuid.uuid4())[:8]
        video_frame_dir = os.path.join(self.output_dir, f"video_{video_id}")
        os.makedirs(video_frame_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = metadata['fps']
        
        extracted_frames = []
        
        for idx, frame_num in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_id = f"{video_id}_frame_{idx:06d}"
            timestamp = frame_num / fps if fps > 0 else 0
            frame_path = os.path.join(video_frame_dir, f"{frame_id}.jpg")
            
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            extracted_frames.append({
                'frame_id': frame_id,
                'frame_path': frame_path,
                'frame_number': frame_num,
                'timestamp': timestamp,
                'video_id': video_id
            })
            
            if progress_callback:
                progress_callback(idx + 1, n_frames)
        
        cap.release()
        
        return extracted_frames
    
    def get_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[Image.Image]:
        """
        Extract a single frame at specific timestamp
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            
        Returns:
            PIL Image or None
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def create_video_thumbnail(self, video_path: str, timestamp: float = 1.0) -> Optional[str]:
        """
        Create a thumbnail image for a video
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds to capture thumbnail (default: 1.0)
            
        Returns:
            Path to thumbnail image or None
        """
        frame = self.get_frame_at_timestamp(video_path, timestamp)
        
        if frame is None:
            return None
        
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        thumbnail_path = os.path.join(self.output_dir, f"{video_id}_thumb.jpg")
        
        thumbnail = frame.resize((320, 180), Image.Resampling.LANCZOS)
        thumbnail.save(thumbnail_path, quality=85)
        
        return thumbnail_path
