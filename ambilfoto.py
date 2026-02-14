"""
AI-Enhanced Image Processing System
- Smart preview generation with VISIBLE watermarks (LOCAL)
- Compressed NON-watermark files to Dropbox (for purchase)
- NO HD files - all compressed
- ALL ENDPOINTS PRESERVED
"""

from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, url_for
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from datetime import datetime
import base64
import io
import ssl
import glob
import json
import uuid
import sys
import socket
import mimetypes
import zipfile
from io import BytesIO
import dropbox
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import WriteMode

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# MySQL Configuration
MYSQL_CONFIG = {
    'host': os.getenv('localhost', '172.28.176.1'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'fotomap_db')
}

def get_db_connection():
    """Get MySQL database connection"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def save_photo_match(user_id, event_photo_id, distance, similarity_score, confidence_score):
    """Save photo match to database"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        check_query = """
            SELECT id FROM user_photo_matches 
            WHERE user_id = %s AND event_photo_id = %s
        """
        cursor.execute(check_query, (user_id, event_photo_id))
        existing = cursor.fetchone()
        
        if existing:
            update_query = """
                UPDATE user_photo_matches 
                SET distance = %s, 
                    similarity_score = %s, 
                    confidence_score = %s,
                    matched_at = NOW()
                WHERE id = %s
            """
            cursor.execute(update_query, (distance, similarity_score, confidence_score, existing[0]))
            print(f"✅ Updated match: {existing[0]}")
        else:
            match_id = str(uuid.uuid4())
            insert_query = """
                INSERT INTO user_photo_matches 
                (id, user_id, event_photo_id, distance, similarity_score, confidence_score, matched_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(insert_query, (match_id, user_id, event_photo_id, distance, similarity_score, confidence_score))
            print(f"✅ Created match: {match_id}")
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
        
    except Error as e:
        print(f"❌ Save match error: {e}")
        if connection:
            connection.close()
        return False

def get_event_photo_id_by_ai_photo_id(ai_photo_id):
    """Get event_photo_id from ai_photo_id"""
    connection = get_db_connection()
    if not connection:
        return None
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT id FROM event_photos WHERE ai_photo_id = %s AND is_deleted = 0"
        cursor.execute(query, (ai_photo_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result:
            return result['id']
        return None
        
    except Error as e:
        print(f"❌ Query error: {e}")
        if connection:
            connection.close()
        return None

# ==================== DROPBOX CONFIGURATION ====================
DROPBOX_ACCESS_TOKEN = "sl.u.AGReoKGXlkG0CdfF3tipnBM-6r8LVNqxIlTFdqnftMfDZmGXrAtGvIYfGcn4_7YowkyN6P7XRj8kuoJ1Wu_8zEjWA1HqZLTwKl9y2N8_JCPZEQ69d-RxSJI6pzZFVmhgJ61z1MAIFywQw2tE2MvXjxfcGLGsp5TwPDcB4_GOJnGtwD1aL9tGvksoXGJWTDqEPu2i-iVYXVcrBiJM6g8cu5djjh7iKB_6JYLNmYU5RWon5jPBmY0mOZUw5u9o84SrII0LLXAKRc_c4G9kXvDDqy6PV8IiSGvH3SF27LjVzF7WTsOt7w28sOG0_Ychnb107owncONAuXvjCylloUaolO20V1NMg_WiwN3M9QWRM_kREUAKSMJN6Ggafv_T9_taiyijcb3VIuVCt0jdKjm4T9bHutInuhQLjVAspsm5kUVo-sM6aNQ-dIR6S0ZPtBbin9gU1y5yg2flhv-uI4bq1jSWMyLQZwe7zxkrgjS_kng4Pv7kHckJlSBTRS37gdq0fZj5K65xCDkpAPzEdTBKXCuEBdfDJquUGU_G6bcwGV4lA1Q2IMoT0DUR7s4jwS5uhwwUp88TE0U61EUHJUpfh8jV4A2Aj7mQU8xRNo3SyiEB8v4rOCzU90f2mca6Qc9rbo5SvvnM8AZ5b2Qc2_b4FwC0ZnsvbF62UG6_g5MZHmzyl1djUCfZKAAkAr8cke0QVADg8n0qwEMUhnhkec9Fuv9jL0Ji_ExU7WuWnIXy1KLFjkhwas0KEbc_OvIcjRPBiUOlVF5xfLoia6sV3BLVdnAqOHfSmWTMbRBfiN5BJ0h8oVDwNp7bOokBrhuBeeMI1pSH4z13te0szKlDCwlCS28prC56NZYrN0BFhP8sUghxcqBjceHYS4iVIY0IvZ6A-0Ku0c8rxAU5-MQEdVAZsSiHt3auj81nNr8Vw1otMOPX5nLzBosKrT5AC0rbBf0h6G-LasQ5tpnIVtmLgKTXdtIl-QQ2-Axd73R_1eFicEjwjUbF803YrVBPLjXgvdwjk4iNdeRrpDSeumuQpM1k2G5dQ4Y3fjAOr3d3v0a34FBKYWb7IkiUAmM3nI9d7Db1--Ik0iZiOo5lyKsDRIUpgJ7Hu3B4goRbG9lDyAUlbz2CBfKC0RTc1tUg2yzo-_tGd8CIuZQodttcjPoOH71-FY3jmlaoPZAOXn68KGHMAzQFc84Zxvbm_qerz_40zuquek42s7sWcMbKnZpIjAgllWQRmW2P4jOibQsXdRZ8PMWae_gkto8giJOALBXTg6sOjd0MgBnuKHUmISfV30709cVqu9trwsJVmhJwOCTN6cEOqkMr9zONUGx_BjZhjuFLX3CZLrEYqt-wsNsPSceumgDCUuht8oxoG1PXgcCDj3TsnAqa0b72Ft_fIUKzT3Vm1_r8m1IxDnkZTyZzTmWktKfBxhdRxl9OThwtgdJsQ7ihOQ"
DROPBOX_FOLDER = "/tes-ambilfoto"

# ==================== STORAGE PATHS ====================
UPLOAD_COMPRESSED = 'uploads/compressed'  # Keep old folder for compatibility
UPLOAD_PREVIEW_WATERMARKED = 'uploads/preview_watermarked'  # New: Preview with watermark
UPLOAD_TEMP = 'uploads/temp'

# Watermark logo path
WATERMARK_LOGO = 'ambilfoto-logo.png'


class SmartImageProcessor:
    """AI-Enhanced Image Processing with dual outputs"""
    
    def __init__(self, watermark_logo_path=None):
        self.watermark_logo = None
        if watermark_logo_path and os.path.exists(watermark_logo_path):
            try:
                self.watermark_logo = Image.open(watermark_logo_path).convert('RGBA')
                print(f"✅ Watermark logo loaded: {watermark_logo_path}")
                print(f"   Size: {self.watermark_logo.size}")
            except Exception as e:
                print(f"⚠️  Failed to load watermark: {e}")
        else:
            print(f"⚠️  Watermark file not found: {watermark_logo_path}")
    
    def add_multi_watermark(self, img, logo_size=180, opacity=0.65, spacing=200):
        """Add multiple HIGHLY VISIBLE watermarks across the image"""
        if not self.watermark_logo:
            print("⚠️  No watermark logo available - skipping watermark")
            return img
        
        try:
            # Convert to RGBA
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create watermark layer
            watermark_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            
            # Resize logo - BIGGER SIZE
            logo = self.watermark_logo.copy()
            logo.thumbnail((logo_size, logo_size), Image.LANCZOS)
            
            # ENHANCE LOGO COLORS for visibility
            if logo.mode == 'RGBA':
                # Split channels
                r, g, b, a = logo.split()
                
                # Merge RGB for enhancement
                rgb = Image.merge('RGB', (r, g, b))
                
                # Increase color saturation 2x
                enhancer = ImageEnhance.Color(rgb)
                rgb = enhancer.enhance(2.0)
                
                # Increase contrast 1.5x
                enhancer = ImageEnhance.Contrast(rgb)
                rgb = enhancer.enhance(1.5)
                
                # Split enhanced RGB
                r, g, b = rgb.split()
                
                # Apply HIGH opacity
                a = a.point(lambda p: int(p * opacity))
                
                # Merge back
                logo = Image.merge('RGBA', (r, g, b, a))
            
            # Get dimensions
            img_width, img_height = img.size
            logo_width, logo_height = logo.size
            
            watermark_count = 0
            
            # Grid pattern with edge coverage
            for y in range(-logo_height//2, img_height + logo_height//2, spacing):
                for x in range(-logo_width//2, img_width + logo_width//2, spacing):
                    # Random offset for natural look
                    offset_x = (x + y) % 40 - 20
                    offset_y = (y + x) % 40 - 20
                    
                    pos_x = x + offset_x
                    pos_y = y + offset_y
                    
                    # Paste watermark
                    try:
                        watermark_layer.paste(logo, (pos_x, pos_y), logo)
                        watermark_count += 1
                    except:
                        pass
            
            print(f"   🔒 Added {watermark_count} VISIBLE watermarks (size: {logo_size}px, opacity: {int(opacity*100)}%)")
            
            # Composite
            watermarked = Image.alpha_composite(img, watermark_layer)
            watermarked = watermarked.convert('RGB')
            
            return watermarked
            
        except Exception as e:
            print(f"⚠️  Watermark error: {e}")
            import traceback
            traceback.print_exc()
            return img.convert('RGB') if img.mode != 'RGB' else img
    
    def create_preview_version(self, img, max_dimension=800, target_size_kb=150, quality_start=35):
        """
        ✅ OLD METHOD - Create preview with VISIBLE watermarks
        Kept for backward compatibility
        """
        try:
            # Convert to RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            # Add VISIBLE watermarks
            print(f"   🔒 Adding VISIBLE watermarks...")
            img = self.add_multi_watermark(img, logo_size=180, opacity=0.65, spacing=200)
            
            # Compress
            output = BytesIO()
            quality = quality_start
            
            while quality >= 15:
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
                
                size_kb = output.tell() / 1024
                if size_kb <= target_size_kb:
                    break
                quality -= 5
            
            final_size = output.tell() / 1024
            print(f"   ✅ Preview created: {final_size:.1f}KB (VISIBLE watermark)")
            
            output.seek(0)
            return output
            
        except Exception as e:
            print(f"❌ Preview creation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_preview_with_watermark(self, img, max_dimension=1280, target_size_kb=150, quality_start=40):
        """
        ✅ NEW: Create LOCAL preview with VISIBLE watermark
        For display purposes only
        """
        try:
            # Convert to RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            # Add VISIBLE watermarks
            print(f"   🔒 Adding VISIBLE watermarks to LOCAL preview...")
            img = self.add_multi_watermark(img, logo_size=180, opacity=0.65, spacing=200)
            
            # Compress
            output = BytesIO()
            quality = quality_start
            
            while quality >= 15:
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
                
                size_kb = output.tell() / 1024
                if size_kb <= target_size_kb:
                    break
                quality -= 5
            
            final_size = output.tell() / 1024
            print(f"   ✅ LOCAL Preview (watermarked): {final_size:.1f}KB")
            
            output.seek(0)
            return output
            
        except Exception as e:
            print(f"❌ Preview creation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_compressed_for_dropbox(self, img, max_dimension=1920, target_size_kb=500, quality_start=75):
        """
        ✅ NEW: Create compressed version WITHOUT watermark for Dropbox
        For download after purchase
        """
        try:
            # Convert to RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Smart resize
            original_size = max(img.size)
            if original_size > max_dimension:
                ratio = max_dimension / original_size
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
                print(f"   📐 Resized to: {img.size[0]}x{img.size[1]}")
            
            # Apply subtle enhancement
            img = self.apply_smart_enhancement(img)
            
            # Compress to target size
            output = BytesIO()
            quality = quality_start
            
            while quality >= 50:
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
                
                size_kb = output.tell() / 1024
                if size_kb <= target_size_kb:
                    break
                quality -= 5
            
            final_size = output.tell() / 1024
            print(f"   💾 Dropbox version (NO watermark): {final_size:.1f}KB")
            
            output.seek(0)
            return output
            
        except Exception as e:
            print(f"❌ Compression error: {e}")
            return None
    
    def create_optimized_original(self, img, max_dimension=2560, target_quality=85):
        """
        ✅ OLD METHOD - Create optimized version for download (no watermark)
        Kept for backward compatibility
        """
        try:
            # Convert to RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Smart resize
            original_size = max(img.size)
            if original_size > max_dimension:
                ratio = max_dimension / original_size
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
                print(f"📐 Resized: {img.size[0]}x{img.size[1]}")
            
            # Apply enhancement
            img = self.apply_smart_enhancement(img)
            
            # Compress
            output = BytesIO()
            img.save(output, format='JPEG', quality=target_quality, optimize=True, progressive=True)
            
            size_kb = output.tell() / 1024
            print(f"💾 Optimized size: {size_kb:.1f}KB")
            
            output.seek(0)
            return output
            
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return None
    
    def create_purchased_version(self, img, max_dimension=1920, target_quality=75):
        """
        ✅ ALIAS for create_compressed_for_dropbox
        Kept for backward compatibility with purchase system
        """
        return self.create_compressed_for_dropbox(img, max_dimension, target_quality * 7, target_quality)
    
    def apply_smart_enhancement(self, img):
        """Apply subtle enhancement"""
        try:
            img_array = np.array(img)
            
            # Subtle sharpening
            kernel = np.array([[-0.5, -0.5, -0.5],
                             [-0.5,  5.0, -0.5],
                             [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(img_array, -1, kernel * 0.1)
            
            # Blend
            enhanced = cv2.addWeighted(img_array, 0.9, sharpened, 0.1, 0)
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            print(f"⚠️  Enhancement skipped: {e}")
            return img


class DropboxManager:
    """Manager untuk semua operasi Dropbox"""
    
    def __init__(self, access_token):
        try:
            if not access_token:
                print("⚠️  No Dropbox token provided")
                self.dbx = None
                return
                
            self.dbx = dropbox.Dropbox(access_token)
            self.dbx.users_get_current_account()
            print("✅ Dropbox connected successfully!")
        except AuthError:
            print("❌ ERROR: Invalid Dropbox access token")
            self.dbx = None
        except Exception as e:
            print(f"❌ Dropbox connection error: {e}")
            self.dbx = None
    
    def upload_from_memory(self, file_bytes, dropbox_path):
        """Upload file from memory to Dropbox"""
        if not self.dbx:
            return {'success': False, 'error': 'Dropbox not connected'}
        
        try:
            self.dbx.files_upload(
                file_bytes,
                dropbox_path,
                mode=WriteMode('overwrite'),
                autorename=False
            )
            
            try:
                shared_link = self.dbx.sharing_create_shared_link_with_settings(dropbox_path)
                download_link = shared_link.url.replace('?dl=0', '?dl=1')
            except ApiError as e:
                if 'shared_link_already_exists' in str(e):
                    links = self.dbx.sharing_list_shared_links(path=dropbox_path).links
                    download_link = links[0].url.replace('?dl=0', '?dl=1') if links else None
                else:
                    download_link = None
            
            print(f"   ✅ Uploaded to Dropbox: {dropbox_path}")
            return {
                'success': True,
                'path': dropbox_path,
                'shared_link': download_link
            }
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_file(self, dropbox_path):
        """Download file from Dropbox"""
        if not self.dbx:
            return None
        
        try:
            metadata, response = self.dbx.files_download(dropbox_path)
            return BytesIO(response.content)
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def get_temporary_link(self, dropbox_path):
        """Get temporary download link"""
        if not self.dbx:
            return None
        
        try:
            result = self.dbx.files_get_temporary_link(dropbox_path)
            return result.link
        except Exception as e:
            print(f"❌ Temp link error: {e}")
            return None
    
    def create_folder(self, folder_path):
        """Create folder in Dropbox"""
        if not self.dbx:
            return False
        
        try:
            self.dbx.files_create_folder_v2(folder_path)
            print(f"📁 Created folder: {folder_path}")
            return True
        except ApiError as e:
            if 'conflict' in str(e).lower():
                return True
            else:
                print(f"❌ Create folder error: {e}")
                return False


class SoccerClinicFaceRecognition:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=40,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709, 
            post_process=True,
            keep_all=True,
            device=self.device
        )
        
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.photo_database = {
            'photos': {},
            'face_embeddings': {}
        }
        
        self.threshold = 0.85
        self.load_database()
    
    def enhance_face_image(self, face_img):
        """Enhance face image quality"""
        face_array = np.array(face_img)
        lab = cv2.cvtColor(face_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return Image.fromarray(enhanced)
    
    def extract_embedding(self, face_img):
        """Extract face embedding"""
        with torch.no_grad():
            face_img = face_img.to(self.device)
            embedding = self.resnet(face_img.unsqueeze(0))
        return embedding.cpu().numpy().flatten()
    
    def detect_faces_from_image_array(self, img):
        """Detect faces from numpy array"""
        height, width = img.shape[:2]
        max_dimension = 1920
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        boxes, probs = self.mtcnn.detect(img_pil)
        
        faces_data = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    expand_x = int(width * 0.2)
                    expand_y = int(height * 0.25)
                    
                    x1 = max(0, x1 - expand_x)
                    y1 = max(0, y1 - expand_y)
                    x2 = min(img.shape[1], x2 + expand_x)
                    y2 = min(img.shape[0], y2 + expand_y)
                    
                    face_img = img_pil.crop((x1, y1, x2, y2))
                    face_img = self.enhance_face_image(face_img)
                    face_img = face_img.resize((160, 160), Image.LANCZOS)
                    
                    face_array = np.array(face_img).astype(np.float32)
                    face_array = (face_array - 127.5) / 128.0
                    face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
                    
                    embedding = self.extract_embedding(face_tensor)
                    embedding_id = str(uuid.uuid4())
                    
                    self.photo_database['face_embeddings'][embedding_id] = embedding.tolist()
                    
                    faces_data.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(probs[i]),
                        'embedding_id': embedding_id
                    })
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
        
        return faces_data
    
    def match_user_face(self, user_embedding):
        """Match user face with photos"""
        matched_photos = []
        user_emb = np.array(user_embedding)
        user_emb = user_emb / np.linalg.norm(user_emb)
        
        for photo_id, photo_data in self.photo_database['photos'].items():
            for face in photo_data.get('faces_data', []):
                embedding_id = face.get('embedding_id')
                if embedding_id in self.photo_database['face_embeddings']:
                    stored_emb = np.array(self.photo_database['face_embeddings'][embedding_id])
                    stored_emb = stored_emb / np.linalg.norm(stored_emb)
                    
                    distance = np.linalg.norm(user_emb - stored_emb)
                    cosine_sim = np.dot(user_emb, stored_emb)
                    
                    if distance < self.threshold and cosine_sim > 0.4:
                        matched_photos.append({
                            'photo_id': photo_id,
                            'photo_data': photo_data,
                            'distance': float(distance),
                            'cosine_similarity': float(cosine_sim)
                        })
                        break
        
        matched_photos.sort(key=lambda x: x['distance'])
        return matched_photos
    
    def save_database(self):
        """Save database to disk"""
        with open('soccer_clinic_db.pkl', 'wb') as f:
            pickle.dump(self.photo_database, f)
    
    def load_database(self):
        """Load database from disk"""
        if os.path.exists('soccer_clinic_db.pkl'):
            with open('soccer_clinic_db.pkl', 'rb') as f:
                self.photo_database = pickle.load(f)
            print(f"✅ Database loaded: {len(self.photo_database['photos'])} photos")
        else:
            print("📂 New database created")


# Initialize systems
face_system = SoccerClinicFaceRecognition()
image_processor = SmartImageProcessor(WATERMARK_LOGO)
dropbox_manager = DropboxManager(DROPBOX_ACCESS_TOKEN)

# Create directories
for directory in [UPLOAD_COMPRESSED, UPLOAD_PREVIEW_WATERMARKED, UPLOAD_TEMP]:
    os.makedirs(directory, exist_ok=True)

if dropbox_manager.dbx:
    dropbox_manager.create_folder(DROPBOX_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/api/photographer/upload', methods=['POST', 'OPTIONS'])
@cross_origin()
def photographer_upload():
    """
    ✅ MODIFIED: Upload photo with dual strategy
    1. Local preview WITH watermark (for browsing)
    2. Dropbox compressed WITHOUT watermark (for purchase)
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        if not data or 'image' not in data or 'filename' not in data:
            return jsonify({
                'success': False,
                'error': 'image and filename are required'
            }), 400
        
        image_data = data['image']
        filename = data['filename']
        metadata = data.get('metadata', {})
        
        complete_metadata = {
            'event_id': metadata.get('event_id', ''),
            'event_name': metadata.get('event_name', 'Untitled Event'),
            'event_type': metadata.get('event_type', 'general'),
            'event_date': metadata.get('event_date', datetime.now().strftime('%Y-%m-%d')),
            'location': metadata.get('location', 'Not specified'),
            'photographer_id': metadata.get('photographer_id', ''),
            'photographer_name': metadata.get('photographer_name', 'Photographer'),
            'uploaded_by': metadata.get('uploaded_by', ''),
            'upload_timestamp': metadata.get('upload_timestamp', datetime.now().isoformat())
        }
        
        print('='*70)
        print('📝 DUAL STRATEGY UPLOAD')
        print('='*70)
        print(f'📸 Filename: {filename}')
        print('='*70)
        
        # Decode image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({'success': False, 'error': 'Invalid image data'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to decode image: {str(e)}'}), 400
        
        # Generate unique filename
        name, ext = os.path.splitext(filename)
        unique_filename = filename
        counter = 1
        
        while any(p['filename'] == unique_filename for p in face_system.photo_database['photos'].values()):
            unique_filename = f"{name}_{counter}{ext}"
            counter += 1
        
        # Face detection
        print('🔍 Detecting faces...')
        faces_data = face_system.detect_faces_from_image_array(img)
        print(f'✅ Detected {len(faces_data)} face(s)')
        
        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # ✅ STEP 1: Create LOCAL preview WITH watermark
        print('🔒 Creating LOCAL preview WITH watermark...')
        preview_io = image_processor.create_preview_with_watermark(
            img_pil.copy(), 
            max_dimension=1280, 
            target_size_kb=150
        )
        
        preview_path = os.path.join(UPLOAD_PREVIEW_WATERMARKED, unique_filename)
        preview_success = False
        
        if preview_io:
            with open(preview_path, 'wb') as f:
                f.write(preview_io.getvalue())
            
            preview_size = os.path.getsize(preview_path) / 1024
            print(f'   ✅ LOCAL preview saved: {preview_size:.1f}KB (WITH watermark)')
            preview_success = True
        else:
            print('   ⚠️  Preview creation failed')
        
        # ✅ STEP 2: Create Dropbox version WITHOUT watermark
        print('☁️  Creating Dropbox version WITHOUT watermark...')
        dropbox_io = image_processor.create_compressed_for_dropbox(
            img_pil.copy(), 
            max_dimension=1920, 
            target_size_kb=500
        )
        
        dropbox_success = False
        dropbox_path = None
        dropbox_link = None
        
        if dropbox_io:
            dropbox_bytes = dropbox_io.getvalue()
            dropbox_path = f"{DROPBOX_FOLDER}/{unique_filename}"
            
            dropbox_result = dropbox_manager.upload_from_memory(dropbox_bytes, dropbox_path)
            
            if dropbox_result['success']:
                dropbox_link = dropbox_result.get('shared_link')
                dropbox_success = True
                print(f'   ✅ Dropbox upload successful (NO watermark)')
            else:
                print(f'   ❌ Dropbox upload failed: {dropbox_result.get("error")}')
        else:
            print('   ⚠️  Dropbox compression failed')
        
        # Also save to old compressed folder for backward compatibility
        compressed_path = os.path.join(UPLOAD_COMPRESSED, unique_filename)
        if preview_io:
            preview_io.seek(0)
            with open(compressed_path, 'wb') as f:
                f.write(preview_io.read())
        
        # Save to database
        photo_id = str(uuid.uuid4())
        face_system.photo_database['photos'][photo_id] = {
            'path_original': None,  # No original HD file
            'path_compressed': compressed_path if preview_success else None,  # For backward compat
            'path_preview': preview_path if preview_success else None,  # New: watermarked preview
            'filename': unique_filename,
            'faces_data': faces_data,
            'metadata': complete_metadata,
            'uploaded_at': datetime.now().isoformat(),
            'dropbox_path': dropbox_path if dropbox_success else None,
            'dropbox_link': dropbox_link,
            'has_watermark': image_processor.watermark_logo is not None,
            'has_watermark_preview': preview_success,
            'storage_strategy': 'dual'  # ✅ Mark as dual strategy
        }
        
        face_system.save_database()
        
        print('='*70)
        print('✅ DUAL STRATEGY UPLOAD COMPLETED')
        print(f'📌 Photo ID: {photo_id}')
        print(f'👤 Faces: {len(faces_data)}')
        print(f'🔒 LOCAL Preview (watermark): {"✓" if preview_success else "✗"}')
        print(f'☁️  Dropbox (no watermark): {"✓" if dropbox_success else "✗"}')
        print('='*70 + '\n')
        
        return jsonify({
            'success': True,
            'photo_id': photo_id,
            'faces_detected': len(faces_data),
            'watermarked': image_processor.watermark_logo is not None,
            'preview_created': preview_success,
            'dropbox_uploaded': dropbox_success,
            'compressed': preview_success,  # For backward compat
            'storage_strategy': 'dual'
        }), 201
        
    except Exception as e:
        print('='*70)
        print('❌ UPLOAD ERROR')
        print(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        print('='*70 + '\n')
        
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/photographer/photos', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_photographer_photos():
    """Get all photos"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photos = []
        for photo_id, photo_data in face_system.photo_database['photos'].items():
            photos.append({
                'photo_id': photo_id,
                'filename': photo_data['filename'],
                'faces_count': len(photo_data.get('faces_data', [])),
                'metadata': photo_data.get('metadata', {}),
                'uploaded_at': photo_data.get('uploaded_at', ''),
                'in_dropbox': photo_data.get('dropbox_path') is not None,
                'has_visible_watermark': photo_data.get('has_watermark', False),
                'has_watermark_preview': photo_data.get('has_watermark_preview', False),
                'storage_strategy': photo_data.get('storage_strategy', 'unknown')
            })
        
        photos.sort(key=lambda x: x['uploaded_at'], reverse=True)
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []})


@app.route('/api/user/register_face', methods=['POST', 'OPTIONS'])
@cross_origin()
def user_register_face():
    """User registers face - FIXED VERSION"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # ✅ FIX 1: Validate input
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        image_data = data['image']
        
        # ✅ FIX 2: Clean base64 string properly
        try:
            # Remove data URI prefix if exists
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Remove any whitespace
            image_data = image_data.strip()
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to decode image - invalid image data'
                }), 400
                
        except Exception as decode_error:
            print(f"❌ Image decode error: {str(decode_error)}")
            return jsonify({
                'success': False,
                'error': f'Image decode error: {str(decode_error)}'
            }), 400
        
        # ✅ FIX 3: Detect faces with proper error handling
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        print(f"🔍 Detecting faces in image {img_pil.size}...")
        boxes, probs = face_system.mtcnn.detect(img_pil)
        
        # ✅ FIX 4: Check if faces detected BEFORE accessing array
        if boxes is None or len(boxes) == 0:
            print("⚠️  No face detected in image")
            return jsonify({
                'success': False,
                'error': 'No face detected in image',
                'faces_detected': 0
            }), 200  # Return 200 but success=false
        
        print(f"✅ Detected {len(boxes)} face(s)")
        
        # ✅ FIX 5: Extract first face with bounds checking
        try:
            box = boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Validate coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Check if face region is valid
            if x2 <= x1 or y2 <= y1:
                return jsonify({
                    'success': False,
                    'error': 'Invalid face region detected'
                }), 200
            
            # Crop and resize face
            face_img = img_pil.crop((x1, y1, x2, y2))
            face_img = face_img.resize((160, 160), Image.LANCZOS)
            
            # Convert to tensor
            face_array = np.array(face_img).astype(np.float32)
            face_array = (face_array - 127.5) / 128.0
            face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
            
            # ✅ FIX 6: Extract embedding with error handling
            embedding = face_system.extract_embedding(face_tensor)
            
            if embedding is None or len(embedding) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Failed to extract face embedding'
                }), 200
            
            print(f"✅ Face embedding extracted: {len(embedding)} dimensions")
            print(f"   Confidence: {float(probs[0]):.2%}")
            
            # ✅ FIX 7: Return proper JSON response
            return jsonify({
                'success': True,
                'embedding': embedding.tolist(),
                'faces_detected': len(boxes),
                'confidence': float(probs[0])
            }), 200
            
        except Exception as extraction_error:
            print(f"❌ Face extraction error: {str(extraction_error)}")
            import traceback
            traceback.print_exc()
            
            return jsonify({
                'success': False,
                'error': f'Face extraction failed: {str(extraction_error)}'
            }), 500
        
    except KeyError as e:
        print(f"❌ Missing required field: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Missing required field: {str(e)}'
        }), 400
        
    except Exception as e:
        print(f"❌ Unexpected error in register_face: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/user/my_photos', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_user_photos():
    """Get user matched photos"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        user_embedding = data.get('embedding')
        user_id = data.get('user_id')
        
        if not user_embedding:
            return jsonify({'success': False, 'error': 'Embedding required'}), 400
        
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID required'}), 400
        
        print(f"🔍 Face matching for user: {user_id}")
        
        matched_photos = face_system.match_user_face(user_embedding)
        
        # Save to database
        saved_count = 0
        for match in matched_photos:
            photo_id = match['photo_id']
            distance = match['distance']
            cosine_sim = match['cosine_similarity']
            
            event_photo_id = get_event_photo_id_by_ai_photo_id(photo_id)
            
            if event_photo_id:
                similarity_score = 1 - distance
                confidence_score = cosine_sim
                
                if save_photo_match(user_id, event_photo_id, distance, similarity_score, confidence_score):
                    saved_count += 1
        
        print(f"✅ Saved {saved_count}/{len(matched_photos)} matches")
        
        photos = []
        for match in matched_photos:
            photo_id = match['photo_id']
            photo_data = match['photo_data']
            
            photos.append({
                'photo_id': photo_id,
                'filename': photo_data['filename'],
                'metadata': photo_data.get('metadata', {}),
                'distance': match['distance'],
                'cosine_similarity': match['cosine_similarity'],
                'in_dropbox': photo_data.get('dropbox_path') is not None,
                'has_visible_watermark': photo_data.get('has_watermark', False),
                'has_watermark_preview': photo_data.get('has_watermark_preview', False),
                'storage_strategy': photo_data.get('storage_strategy', 'unknown')
            })
        
        return jsonify({
            'success': True, 
            'photos': photos,
            'matches_saved_to_db': saved_count
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e), 'photos': []}), 500


@app.route('/api/image/preview/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def serve_preview_image(photo_id):
    """
    ✅ Serve WATERMARKED preview (backward compatible)
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'error': 'Photo not found'}), 404
        
        # Try new preview path first
        preview_path = photo_data.get('path_preview')
        if preview_path and os.path.exists(preview_path):
            return send_file(
                preview_path,
                mimetype='image/jpeg',
                as_attachment=False
            )
        
        # Fallback to old compressed path
        compressed_path = photo_data.get('path_compressed')
        if compressed_path and os.path.exists(compressed_path):
            return send_file(
                compressed_path,
                mimetype='image/jpeg',
                as_attachment=False
            )
        
        return jsonify({'error': 'Preview not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/dropbox/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_from_dropbox(photo_id):
    """
    ✅ KEPT: Download compressed photo WITHOUT watermark from Dropbox
    For backward compatibility
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        dropbox_path = photo_data.get('dropbox_path')
        if not dropbox_path:
            return jsonify({'success': False, 'error': 'File not in Dropbox'}), 404
        
        if not dropbox_manager.dbx:
            return jsonify({'success': False, 'error': 'Dropbox not available'}), 503
        
        try:
            temp_link = dropbox_manager.get_temporary_link(dropbox_path)
            
            if temp_link:
                return redirect(temp_link)
            else:
                raise Exception("Failed to generate Dropbox link")
                
        except Exception as dropbox_error:
            file_stream = dropbox_manager.download_file(dropbox_path)
            
            if file_stream:
                filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
                return send_file(
                    file_stream,
                    mimetype='image/jpeg',
                    as_attachment=True,
                    download_name=filename
                )
            else:
                raise Exception("Failed to download from Dropbox")
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'photo_id': photo_id
        }), 500


@app.route('/api/photo/download_direct/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_photo_direct(photo_id):
    """
    ✅ KEPT: Download photo directly as binary (for hi-res purchased photos)
    This endpoint sends the actual file bytes, not a redirect
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Get photo data from database
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({
                'success': False, 
                'error': 'Photo not found'
            }), 404
        
        dropbox_path = photo_data.get('dropbox_path')
        if not dropbox_path:
            return jsonify({
                'success': False, 
                'error': 'File not in Dropbox'
            }), 404
        
        if not dropbox_manager.dbx:
            return jsonify({
                'success': False, 
                'error': 'Dropbox not available'
            }), 503
        
        print(f'📥 Direct download request for: {photo_id}')
        print(f'   Dropbox path: {dropbox_path}')
        
        # ✅ Download file from Dropbox to memory
        try:
            metadata, response = dropbox_manager.dbx.files_download(dropbox_path)
            file_bytes = response.content
            
            print(f'✅ Downloaded from Dropbox: {len(file_bytes) / 1024 / 1024:.2f}MB')
            
            # Determine filename
            filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
            
            # ✅ Send file as binary with proper headers
            return send_file(
                BytesIO(file_bytes),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=filename,
                max_age=0  # No caching
            )
            
        except dropbox.exceptions.ApiError as e:
            error_msg = str(e)
            print(f'❌ Dropbox API error: {error_msg}')
            
            if 'not_found' in error_msg.lower():
                return jsonify({
                    'success': False,
                    'error': 'File not found in Dropbox',
                    'dropbox_path': dropbox_path
                }), 404
            else:
                return jsonify({
                    'success': False,
                    'error': f'Dropbox error: {error_msg}'
                }), 500
        
    except Exception as e:
        print(f'❌ Download error: {str(e)}')
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'photo_id': photo_id
        }), 500


@app.route('/api/photo/get_base64/<photo_id>', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_photo_base64(photo_id):
    """
    ✅ KEPT: Get photo as base64 (if direct binary doesn't work)
    This returns JSON with base64 string - less efficient but more compatible
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({
                'success': False,
                'error': 'Photo not found'
            }), 404
        
        dropbox_path = photo_data.get('dropbox_path')
        if not dropbox_path or not dropbox_manager.dbx:
            return jsonify({
                'success': False,
                'error': 'File not available'
            }), 404
        
        print(f'📥 Base64 request for: {photo_id}')
        
        # Download from Dropbox
        metadata, response = dropbox_manager.dbx.files_download(dropbox_path)
        file_bytes = response.content
        
        # Convert to base64
        base64_data = base64.b64encode(file_bytes).decode('utf-8')
        
        # Add data URL prefix
        image_data_url = f'data:image/jpeg;base64,{base64_data}'
        
        print(f'✅ Converted to base64: {len(base64_data) / 1024:.1f}KB')
        
        return jsonify({
            'success': True,
            'photo_id': photo_id,
            'filename': photo_data.get('filename'),
            'face_image': image_data_url,  # ✅ Full data URL format
            'metadata': photo_data.get('metadata', {})
        })
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/photo/generate_optimized/<photo_id>', methods=['POST', 'OPTIONS'])
@cross_origin()
def generate_optimized_version(photo_id):
    """
    ✅ KEPT: Generate optimized version for purchased photo
    Called by Node.js after successful payment
    NOTE: In dual strategy, this is mostly redundant since Dropbox already has compressed version
    But kept for backward compatibility
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({
                'success': False,
                'error': 'Photo not found'
            }), 404
        
        dropbox_path = photo_data.get('dropbox_path')
        
        # Check if already has optimized version
        if photo_data.get('optimized_path'):
            print(f'ℹ️  Already has optimized version: {photo_data["optimized_path"]}')
            return jsonify({
                'success': True,
                'optimized_path': photo_data['optimized_path'],
                'optimized_link': photo_data.get('optimized_link'),
                'already_exists': True
            })
        
        if not dropbox_path or not dropbox_manager.dbx:
            return jsonify({
                'success': False,
                'error': 'Original file not available'
            }), 404
        
        print(f'🎨 Generating optimized version for: {photo_id}')
        
        # Download from Dropbox
        metadata, response = dropbox_manager.dbx.files_download(dropbox_path)
        image_bytes = response.content
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Create optimized version (NO WATERMARK)
        optimized_io = image_processor.create_purchased_version(
            img_pil, 
            max_dimension=1920, 
            target_quality=75
        )
        
        if not optimized_io:
            return jsonify({
                'success': False,
                'error': 'Failed to create optimized version'
            }), 500
        
        # Upload to Dropbox with prefix
        filename = photo_data.get('filename')
        optimized_path = f"{DROPBOX_FOLDER}/optimized_{filename}"
        
        optimized_bytes = optimized_io.getvalue()
        upload_result = dropbox_manager.upload_from_memory(
            optimized_bytes, 
            optimized_path
        )
        
        if not upload_result['success']:
            return jsonify({
                'success': False,
                'error': 'Failed to upload optimized version'
            }), 500
        
        # Update photo database
        photo_data['optimized_path'] = optimized_path
        photo_data['optimized_link'] = upload_result.get('shared_link')
        face_system.save_database()
        
        print(f'✅ Optimized version created: {optimized_path}')
        print(f'   Size: {len(optimized_bytes) / 1024:.1f}KB')
        
        return jsonify({
            'success': True,
            'optimized_path': optimized_path,
            'optimized_link': upload_result.get('shared_link'),
            'size_kb': len(optimized_bytes) / 1024
        })
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/photo/download_optimized/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_optimized(photo_id):
    """
    ✅ KEPT: Download optimized version (for purchased photos)
    - NO watermark
    - Compressed
    - Good quality
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({
                'success': False,
                'error': 'Photo not found'
            }), 404
        
        optimized_path = photo_data.get('optimized_path')
        if not optimized_path:
            # Fallback to regular dropbox path (which is now also compressed without watermark)
            print('ℹ️  No separate optimized version, using main Dropbox file')
            return redirect(url_for('download_from_dropbox', photo_id=photo_id))
        
        if not dropbox_manager.dbx:
            return jsonify({
                'success': False,
                'error': 'Dropbox not available'
            }), 503
        
        print(f'📥 Downloading optimized: {photo_id}')
        
        # Download optimized from Dropbox
        try:
            metadata, response = dropbox_manager.dbx.files_download(optimized_path)
            file_bytes = response.content
            
            print(f'✅ Downloaded: {len(file_bytes) / 1024:.1f}KB (optimized, no watermark)')
            
            filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
            
            return send_file(
                BytesIO(file_bytes),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=filename,
                max_age=0
            )
            
        except dropbox.exceptions.ApiError as e:
            print(f'❌ Dropbox error: {str(e)}')
            
            # Fallback to regular download
            return redirect(url_for('download_from_dropbox', photo_id=photo_id))
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download/purchased/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_purchased_photo(photo_id):
    """
    ✅ NEW: Unified endpoint for downloading purchased photos
    Downloads compressed photo WITHOUT watermark from Dropbox
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        # Check for optimized version first
        optimized_path = photo_data.get('optimized_path')
        if optimized_path and dropbox_manager.dbx:
            try:
                metadata, response = dropbox_manager.dbx.files_download(optimized_path)
                file_bytes = response.content
                filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
                
                print(f'✅ Downloaded optimized: {len(file_bytes) / 1024:.1f}KB')
                
                return send_file(
                    BytesIO(file_bytes),
                    mimetype='image/jpeg',
                    as_attachment=True,
                    download_name=filename,
                    max_age=0
                )
            except:
                pass
        
        # Fallback to main dropbox path
        dropbox_path = photo_data.get('dropbox_path')
        if not dropbox_path:
            return jsonify({'success': False, 'error': 'File not in Dropbox'}), 404
        
        if not dropbox_manager.dbx:
            return jsonify({'success': False, 'error': 'Dropbox not available'}), 503
        
        print(f'📥 Downloading purchased photo: {photo_id}')
        
        try:
            metadata, response = dropbox_manager.dbx.files_download(dropbox_path)
            file_bytes = response.content
            
            print(f'✅ Downloaded: {len(file_bytes) / 1024:.1f}KB (NO watermark)')
            
            filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
            
            return send_file(
                BytesIO(file_bytes),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=filename,
                max_age=0
            )
            
        except dropbox.exceptions.ApiError as e:
            return jsonify({
                'success': False,
                'error': f'Download failed: {str(e)}'
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/photographer/delete/<photo_id>', methods=['DELETE', 'OPTIONS'])
@cross_origin()
def delete_photo(photo_id):
    """Delete photo"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        # Delete from Dropbox
        dropbox_deleted = False
        if photo_data.get('dropbox_path') and dropbox_manager.dbx:
            try:
                dropbox_manager.dbx.files_delete_v2(photo_data['dropbox_path'])
                dropbox_deleted = True
            except:
                pass
        
        # Delete optimized from Dropbox
        optimized_deleted = False
        if photo_data.get('optimized_path') and dropbox_manager.dbx:
            try:
                dropbox_manager.dbx.files_delete_v2(photo_data['optimized_path'])
                optimized_deleted = True
            except:
                pass
        
        # Delete preview
        preview_deleted = False
        if photo_data.get('path_preview') and os.path.exists(photo_data['path_preview']):
            try:
                os.remove(photo_data['path_preview'])
                preview_deleted = True
            except:
                pass
        
        # Delete compressed (backward compat)
        compressed_deleted = False
        if photo_data.get('path_compressed') and os.path.exists(photo_data['path_compressed']):
            try:
                os.remove(photo_data['path_compressed'])
                compressed_deleted = True
            except:
                pass
        
        # Delete embeddings
        faces_data = photo_data.get('faces_data', [])
        for face in faces_data:
            embedding_id = face.get('embedding_id')
            if embedding_id in face_system.photo_database['face_embeddings']:
                del face_system.photo_database['face_embeddings'][embedding_id]
        
        # Delete from database
        del face_system.photo_database['photos'][photo_id]
        face_system.save_database()
        
        return jsonify({
            'success': True,
            'message': 'Photo deleted',
            'dropbox_deleted': dropbox_deleted,
            'optimized_deleted': optimized_deleted,
            'preview_deleted': preview_deleted,
            'compressed_deleted': compressed_deleted
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'service': 'AI Face Recognition - Dual Strategy (ALL ENDPOINTS)',
        'total_photos': len(face_system.photo_database['photos']),
        'total_faces': len(face_system.photo_database['face_embeddings']),
        'dropbox_connected': dropbox_manager.dbx is not None,
        'watermark_available': image_processor.watermark_logo is not None,
        'storage_strategy': {
            'preview': 'LOCAL with VISIBLE watermark',
            'dropbox': 'COMPRESSED without watermark',
            'hd_files': 'NONE - all compressed'
        },
        'endpoints': {
            'new': [
                '/api/download/purchased/<id>',
            ],
            'kept': [
                '/api/download/dropbox/<id>',
                '/api/photo/download_direct/<id>',
                '/api/photo/get_base64/<id>',
                '/api/photo/generate_optimized/<id>',
                '/api/photo/download_optimized/<id>',
            ],
            'total': 'ALL ENDPOINTS PRESERVED'
        }
    })


# ==================== BATCH PROCESSING ====================

def batch_process_from_folder(folder_path):
    """Batch process images with dual strategy"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    metadata = {
        'event_name': input("Event Name: ") or 'Batch Upload',
        'location': input("Location: ") or 'Unknown',
        'photographer': input("Photographer: ") or 'Anonymous',
        'date': datetime.now().isoformat()
    }
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = set()
    
    for ext in image_extensions:
        files = glob.glob(os.path.join(folder_path, ext))
        image_files.update(files)
    
    image_files = sorted(list(image_files))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    total = len(image_files)
    print(f"\n{'='*70}")
    print(f"⚽ DUAL STRATEGY BATCH PROCESSING")
    print(f"{'='*70}")
    print(f"📁 Folder: {folder_path}")
    print(f"📸 Total Images: {total}")
    print(f"🎯 Strategy:")
    print(f"   🔒 Preview: WITH watermark (local)")
    print(f"   ☁️  Dropbox: WITHOUT watermark (compressed)")
    print(f"{'='*70}\n")
    
    confirm = input(f"Process {total} images? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Cancelled")
        return
    
    processed = 0
    failed = 0
    uploaded_to_dropbox = 0
    watermarked_count = 0
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            filename = os.path.basename(img_path)
            print(f"[{idx}/{total}] Processing {filename}... ", end='', flush=True)
            
            img = cv2.imread(img_path)
            if img is None:
                print("❌ Failed to read")
                failed += 1
                continue
            
            # Generate unique filename
            name, ext = os.path.splitext(filename)
            unique_filename = filename
            counter = 1
            
            while any(p['filename'] == unique_filename for p in face_system.photo_database['photos'].values()):
                unique_filename = f"{name}_{counter}{ext}"
                counter += 1
            
            # Face detection
            faces_data = face_system.detect_faces_from_image_array(img)
            
            # Convert to PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Create preview WITH watermark
            preview_io = image_processor.create_preview_with_watermark(img_pil.copy())
            preview_path = os.path.join(UPLOAD_PREVIEW_WATERMARKED, unique_filename)
            
            if preview_io:
                with open(preview_path, 'wb') as f:
                    f.write(preview_io.getvalue())
                if image_processor.watermark_logo:
                    watermarked_count += 1
            
            # Create Dropbox version WITHOUT watermark
            dropbox_io = image_processor.create_compressed_for_dropbox(img_pil.copy())
            dropbox_path = f"{DROPBOX_FOLDER}/{unique_filename}"
            
            if dropbox_io:
                dropbox_bytes = dropbox_io.getvalue()
                dropbox_result = dropbox_manager.upload_from_memory(dropbox_bytes, dropbox_path)
                if dropbox_result['success']:
                    uploaded_to_dropbox += 1
            
            # Save to database
            photo_id = str(uuid.uuid4())
            face_system.photo_database['photos'][photo_id] = {
                'path_original': None,
                'path_compressed': preview_path,
                'path_preview': preview_path,
                'filename': unique_filename,
                'faces_data': faces_data,
                'metadata': metadata.copy(),
                'uploaded_at': datetime.now().isoformat(),
                'dropbox_path': dropbox_path,
                'dropbox_link': dropbox_result.get('shared_link', None) if dropbox_io else None,
                'has_watermark': image_processor.watermark_logo is not None,
                'has_watermark_preview': True,
                'storage_strategy': 'dual'
            }
            
            processed += 1
            print(f"✅ {len(faces_data)} faces | ☁️ Dropbox | 🔒 Watermark")
            
            # Checkpoint every 10 images
            if processed % 10 == 0:
                face_system.save_database()
                print(f"💾 Checkpoint saved ({processed}/{total})")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed += 1
            continue
    
    # Final save
    face_system.save_database()
    
    print(f"\n{'='*70}")
    print(f"✅ DUAL STRATEGY BATCH COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Processed: {processed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"☁️  Uploaded to Dropbox: {uploaded_to_dropbox}/{processed}")
    print(f"🔒 Watermarked Preview: {watermarked_count}/{processed}")
    print(f"💾 Database saved")
    print(f"{'='*70}\n")


def generate_self_signed_cert():
    """Generate self-signed certificate for HTTPS"""
    try:
        from OpenSSL import crypto
    except ImportError:
        print("⚠️  PyOpenSSL not installed. Install: pip install pyopenssl")
        return False
    
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("✅ SSL Certificate exists")
        return True
    
    print("🔐 Generating self-signed SSL certificate...")
    
    try:
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        
        cert = crypto.X509()
        cert.get_subject().C = "ID"
        cert.get_subject().ST = "East Kalimantan"
        cert.get_subject().L = "Samarinda"
        cert.get_subject().O = "Soccer Clinic"
        cert.get_subject().OU = "AI Face Recognition"
        cert.get_subject().CN = "localhost"
        
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        
        with open("cert.pem", "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        
        with open("key.pem", "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
        
        print("✅ SSL Certificate created")
        return True
    except Exception as e:
        print(f"❌ Error generating certificate: {e}")
        return False


# ==================== MAIN ====================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch' and len(sys.argv) > 2:
            folder_path = sys.argv[2]
            print("\n" + "="*70)
            print("⚽ DUAL STRATEGY BATCH PROCESSING")
            print("="*70)
            batch_process_from_folder(folder_path)
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("\n" + "="*70)
            print("⚽ AI FACE RECOGNITION - DUAL STRATEGY (ALL ENDPOINTS)")
            print("="*70)
            print("\n📖 USAGE:")
            print("  python app.py                    # Run web server")
            print("  python app.py --batch <folder>   # Batch process with dual strategy")
            print("  python app.py --help             # Show this help")
            print("\n🎯 DUAL STRATEGY:")
            print("  • Preview (LOCAL): WITH watermark, compressed (~150KB)")
            print("  • Dropbox: WITHOUT watermark, compressed (~500KB)")
            print("  • NO HD files - save 90% storage costs")
            print("\n📡 ALL ENDPOINTS PRESERVED:")
            print("  ✅ /api/download/dropbox/<id>")
            print("  ✅ /api/photo/download_direct/<id>")
            print("  ✅ /api/photo/get_base64/<id>")
            print("  ✅ /api/photo/generate_optimized/<id>")
            print("  ✅ /api/photo/download_optimized/<id>")
            print("  🆕 /api/download/purchased/<id> (unified)")
            print("="*70 + "\n")
            sys.exit(0)
        else:
            print("❌ Invalid arguments. Use --help for usage information.")
            sys.exit(1)
    
    if not image_processor.watermark_logo:
        print("\n" + "="*70)
        print("⚠️  WARNING: WATERMARK LOGO NOT FOUND")
        print("="*70)
        print(f"Place 'ambilfoto-logo.png' in the same directory")
        print("Preview images will be created without watermarks.")
        print("="*70 + "\n")
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*70)
    print("⚽ AI FACE RECOGNITION - DUAL STRATEGY (ALL ENDPOINTS)")
    print("="*70)
    print(f"\n☁️  Dropbox Status: {'✅ Connected' if dropbox_manager.dbx else '❌ Disconnected'}")
    print(f"🔒 Watermark Status: {'✅ Available' if image_processor.watermark_logo else '❌ Not Found'}")
    print(f"\n💾 STORAGE STRATEGY:")
    print(f"   🔒 Preview (LOCAL): WITH watermark, ~150KB")
    print(f"   ☁️  Dropbox: WITHOUT watermark, ~500KB")
    print(f"   ❌ HD Files: NONE - save 90% costs")
    print(f"\n📡 ENDPOINTS: ALL PRESERVED + NEW")
    
    use_https = input("\nUse HTTPS? (y/n) [recommended for mobile camera]: ").lower() == 'y'
    
    if use_https:
        has_ssl = generate_self_signed_cert()
        
        if has_ssl:
            print(f"\n🌐 HTTPS URLs:")
            print(f"   Laptop/Desktop: https://localhost:4000")
            print(f"   Mobile (WiFi):  https://{local_ip}:4000")
            print("\n" + "="*70)
            print("✨ DUAL STRATEGY ACTIVE + ALL ENDPOINTS")
            print("="*70 + "\n")
            
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain('cert.pem', 'key.pem')
            
            app.run(host='0.0.0.0', port=4000, debug=False, threaded=True, ssl_context=context)
        else:
            use_https = False
    
    if not use_https:
        print(f"\n🌐 HTTP URLs:")
        print(f"   Laptop/Desktop: http://localhost:4000")
        print(f"   Mobile (WiFi):  http://{local_ip}:4000")
        print("\n" + "="*70)
        print("✨ DUAL STRATEGY ACTIVE + ALL ENDPOINTS")
        print("="*70 + "\n")
        
        app.run(host='0.0.0.0', port=4000, debug=False, threaded=True)