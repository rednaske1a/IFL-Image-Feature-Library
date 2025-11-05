import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

class MetadataDatabase:
    def __init__(self, db_path="./data/metadata.db"):
        """Initialize SQLite database for metadata, tags, ratings, and favorites"""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _get_connection(self):
        """Get or create database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT UNIQUE NOT NULL,
            media_type TEXT NOT NULL,
            source_path TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            rating TEXT DEFAULT 'safe',
            score INTEGER DEFAULT 0,
            view_count INTEGER DEFAULT 0,
            favorite_count INTEGER DEFAULT 0
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            media_id TEXT NOT NULL,
            duration REAL,
            fps REAL,
            total_frames INTEGER,
            codec TEXT,
            FOREIGN KEY (media_id) REFERENCES media(media_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id TEXT UNIQUE NOT NULL,
            video_id TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            timestamp REAL,
            media_id TEXT NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos(video_id),
            FOREIGN KEY (media_id) REFERENCES media(media_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id TEXT UNIQUE NOT NULL,
            media_id TEXT NOT NULL,
            frame_id TEXT,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_w INTEGER,
            bbox_h INTEGER,
            area REAL,
            iou_score REAL,
            stability_score REAL,
            FOREIGN KEY (media_id) REFERENCES media(media_id),
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tag_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            color TEXT,
            description TEXT
        )
        ''')
        
        cursor.execute('''
        INSERT OR IGNORE INTO tag_categories (name, color, description) VALUES
        ('general', '#3b82f6', 'General descriptive tags'),
        ('character', '#8b5cf6', 'Character names'),
        ('artist', '#ec4899', 'Artist or creator'),
        ('meta', '#10b981', 'Metadata tags (resolution, format, etc.)')
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_name TEXT UNIQUE NOT NULL,
            category_id INTEGER,
            usage_count INTEGER DEFAULT 0,
            FOREIGN KEY (category_id) REFERENCES tag_categories(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media(media_id),
            FOREIGN KEY (tag_id) REFERENCES tags(id),
            UNIQUE(media_id, tag_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT NOT NULL,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media(media_id),
            UNIQUE(media_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT NOT NULL,
            rating_value INTEGER CHECK(rating_value >= 1 AND rating_value <= 5),
            rated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media(media_id),
            UNIQUE(media_id)
        )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_rating ON media(rating)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_score ON media(score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_upload_date ON media(upload_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(tag_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_tags_media ON media_tags(media_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_tags_tag ON media_tags(tag_id)')
        
        conn.commit()
    
    def add_media(self, media_id: str, media_type: str, source_path: str,
                  width: int = None, height: int = None, file_size: int = None,
                  rating: str = 'safe') -> int:
        """Add new media entry"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO media (media_id, media_type, source_path, width, height, file_size, rating)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (media_id, media_type, source_path, width, height, file_size, rating))
        conn.commit()
        return cursor.lastrowid
    
    def add_video(self, video_id: str, media_id: str, duration: float,
                  fps: float, total_frames: int, codec: str = None):
        """Add video metadata"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO videos (video_id, media_id, duration, fps, total_frames, codec)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (video_id, media_id, duration, fps, total_frames, codec))
        conn.commit()
        return cursor.lastrowid
    
    def add_frame(self, frame_id: str, video_id: str, frame_number: int,
                  timestamp: float, media_id: str):
        """Add frame metadata"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO frames (frame_id, video_id, frame_number, timestamp, media_id)
        VALUES (?, ?, ?, ?, ?)
        ''', (frame_id, video_id, frame_number, timestamp, media_id))
        conn.commit()
        return cursor.lastrowid
    
    def add_segment(self, segment_id: str, media_id: str, frame_id: str = None,
                    bbox: Tuple[int, int, int, int] = None, area: float = None,
                    iou_score: float = None, stability_score: float = None):
        """Add segment metadata"""
        conn = self._get_connection()
        cursor = conn.cursor()
        bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (None, None, None, None)
        cursor.execute('''
        INSERT OR IGNORE INTO segments 
        (segment_id, media_id, frame_id, bbox_x, bbox_y, bbox_w, bbox_h, area, iou_score, stability_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (segment_id, media_id, frame_id, bbox_x, bbox_y, bbox_w, bbox_h, area, iou_score, stability_score))
        conn.commit()
        return cursor.lastrowid
    
    def add_tag(self, tag_name: str, category: str = 'general') -> int:
        """Add or get tag"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM tag_categories WHERE name = ?', (category,))
        cat_row = cursor.fetchone()
        category_id = cat_row[0] if cat_row else 1
        
        cursor.execute('SELECT id FROM tags WHERE tag_name = ?', (tag_name.lower(),))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor.execute('INSERT INTO tags (tag_name, category_id) VALUES (?, ?)',
                      (tag_name.lower(), category_id))
        conn.commit()
        return cursor.lastrowid
    
    def add_media_tag(self, media_id: str, tag_name: str, category: str = 'general'):
        """Associate tag with media"""
        tag_id = self.add_tag(tag_name, category)
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT OR IGNORE INTO media_tags (media_id, tag_id) VALUES (?, ?)',
                          (media_id, tag_id))
            cursor.execute('UPDATE tags SET usage_count = usage_count + 1 WHERE id = ?', (tag_id,))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
    
    def remove_media_tag(self, media_id: str, tag_name: str):
        """Remove tag from media"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        DELETE FROM media_tags 
        WHERE media_id = ? AND tag_id = (SELECT id FROM tags WHERE tag_name = ?)
        ''', (media_id, tag_name.lower()))
        cursor.execute('''
        UPDATE tags SET usage_count = usage_count - 1 
        WHERE tag_name = ? AND usage_count > 0
        ''', (tag_name.lower(),))
        conn.commit()
    
    def get_media_tags(self, media_id: str) -> List[Dict]:
        """Get all tags for a media item"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT t.tag_name, tc.name as category, tc.color
        FROM media_tags mt
        JOIN tags t ON mt.tag_id = t.id
        JOIN tag_categories tc ON t.category_id = tc.id
        WHERE mt.media_id = ?
        ORDER BY tc.id, t.tag_name
        ''', (media_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def search_by_tags(self, tags: List[str], exclude_tags: List[str] = None,
                       rating: str = None, limit: int = 50) -> List[str]:
        """Search media by tags"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = '''
        SELECT DISTINCT m.media_id, m.score, m.upload_date
        FROM media m
        '''
        
        conditions = []
        params = []
        
        if tags:
            query += '''
            JOIN media_tags mt ON m.media_id = mt.media_id
            JOIN tags t ON mt.tag_id = t.id
            '''
            tag_placeholders = ','.join(['?' for _ in tags])
            conditions.append(f't.tag_name IN ({tag_placeholders})')
            params.extend([tag.lower() for tag in tags])
        
        if exclude_tags:
            exclude_placeholders = ','.join(['?' for _ in exclude_tags])
            conditions.append(f'''
            m.media_id NOT IN (
                SELECT media_id FROM media_tags 
                WHERE tag_id IN (SELECT id FROM tags WHERE tag_name IN ({exclude_placeholders}))
            )
            ''')
            params.extend([tag.lower() for tag in exclude_tags])
        
        if rating:
            conditions.append('m.rating = ?')
            params.append(rating)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY m.score DESC, m.upload_date DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]
    
    def get_popular_tags(self, limit: int = 50) -> List[Dict]:
        """Get most popular tags"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT t.tag_name, tc.name as category, tc.color, t.usage_count
        FROM tags t
        JOIN tag_categories tc ON t.category_id = tc.id
        WHERE t.usage_count > 0
        ORDER BY t.usage_count DESC
        LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def autocomplete_tags(self, prefix: str, limit: int = 10) -> List[str]:
        """Get tag suggestions for autocomplete"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT tag_name FROM tags
        WHERE tag_name LIKE ?
        ORDER BY usage_count DESC
        LIMIT ?
        ''', (f'{prefix.lower()}%', limit))
        return [row[0] for row in cursor.fetchall()]
    
    def add_favorite(self, media_id: str):
        """Add media to favorites"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO favorites (media_id) VALUES (?)', (media_id,))
        cursor.execute('UPDATE media SET favorite_count = favorite_count + 1 WHERE media_id = ?', (media_id,))
        conn.commit()
    
    def remove_favorite(self, media_id: str):
        """Remove media from favorites"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM favorites WHERE media_id = ?', (media_id,))
        cursor.execute('''
        UPDATE media SET favorite_count = favorite_count - 1 
        WHERE media_id = ? AND favorite_count > 0
        ''', (media_id,))
        conn.commit()
    
    def is_favorite(self, media_id: str) -> bool:
        """Check if media is favorited"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM favorites WHERE media_id = ?', (media_id,))
        return cursor.fetchone() is not None
    
    def get_favorites(self, limit: int = 100) -> List[str]:
        """Get all favorited media IDs"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT media_id FROM favorites
        ORDER BY added_date DESC
        LIMIT ?
        ''', (limit,))
        return [row[0] for row in cursor.fetchall()]
    
    def update_rating(self, media_id: str, rating: str):
        """Update content rating (safe, questionable, explicit)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE media SET rating = ? WHERE media_id = ?', (rating, media_id))
        conn.commit()
    
    def update_score(self, media_id: str, score: int):
        """Update media score"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE media SET score = ? WHERE media_id = ?', (score, media_id))
        conn.commit()
    
    def increment_view_count(self, media_id: str):
        """Increment view count"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE media SET view_count = view_count + 1 WHERE media_id = ?', (media_id,))
        conn.commit()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM media')
        total_media = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM videos')
        total_videos = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM segments')
        total_segments = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM tags')
        total_tags = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM favorites')
        total_favorites = cursor.fetchone()[0]
        
        return {
            'total_media': total_media,
            'total_videos': total_videos,
            'total_segments': total_segments,
            'total_tags': total_tags,
            'total_favorites': total_favorites
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
