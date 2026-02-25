"""
AI-Enhanced Image Processing System
- Smart preview generation with VISIBLE watermarks (LOCAL)
- Compressed NON-watermark files to Dropbox (for purchase)
- Compressed NON-watermark files to Biznet NOS (S3-compatible)
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
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config

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
DROPBOX_ACCESS_TOKEN = "sl.u.AGVMUgEBwZX4Gpjad9ECUqlVRWD1YBEPcCXnFBw0TJ02hDV_M9_I_ndeIQRIN5RWZbp7JU1Xc2HQ9DBqDXluTFzwYrmtiU7ZInGTfBQgz907t80Mti9FVU6lCzrjYCE8aiQ8xAbhZMScZX0jsmkuLGh6gLJUeTHWWKE5rxkzwltdJcfQFABE6cf96O4JyrXYRpGvKRDrIDgefzoIeow-jlgcwJGtLAaMAPwBgSyo_gZMltM7JsDq55oxfd5PkjOAzbZ8GC3xw-X8HjA9CJqAZ60aWzgedrBNfeeQBM6xb1DXsSPJc3F3mfHWUSqx3kbv68KYiBO1Zpisi7QWSy3xBYhm3oxHQevYkZx3zPymjoJPGdO4Jj_j5i4K65kIWGv4nd7dv1kOV9UP7okMbfHQdz_lvijPFolkxUZIuURJs_6YMxSVWEa4U7qn5US9XO0bOvJZPg8rkusss03-4jJrMGln7k5675aUH0ADRAUb90VnIhH8DkaGZVfj1Q-DHJRMY63da1o0nHV0_tgfvymySRK4IW5RUQwyH5LU0BkmST7Fl5H2NXI_Ge6ETmWhN2Nba7gMiH3Kli7UmAxFM8krZ-9Uhr_w5amT0Wf2OSkhXvpx_CdsJJ7SO22tYjUKQKqwRz38mw6rgNJhBVe4GsA2m1WZ26XF4foIcMWjM4c82DTwG2R2ypJcg0ACSPseUWJncKus2IZcmxUlc8ph5jqML3Xf57CGzd98n4MtBRS5dkwxynOPsBakwinAEPbmPkBB9CUgsgA5N03ekA4DAPO0_ifP8cBJwfuQmbLVJyoH3VUUdgVeucFPo0-h5lQlmXKizFCc19_ooPglTUL5n3YKfB6e5qvSOin6p5EDFrVj0auDr1vlN-VKs7eLXfqWbolKvftlV63uMq_Ba0DRbOP8nWo4zqV_oWr0mbmRHTJMob6Pl-rUjJtV0ET0b6ZPMCh92xdVSoSngKc_tda1HpAXM-i0r9iLCPnELLUUGtK5OWPT_kwYm6Y1wJJ_vhaD4GN5twMwrdntf4K1si1_9j_BoJvIEmD1qUFSd2IatFIXDjm-dgEWkrsFHT1-YoxARVuNX27KV1VWpmd3iCEAGmiD_FN76VPxotmGGWfIi4dmk5k8IuesXFIe2akuLZvuU1RgR8CvchMnRSKYYbxLw8NTs3UdLmWDHClXoz6hWy_PS39oIQ5xnkmqHzr8TCBZXVo2N027aENapguK88otE6240Mzp0dXQWCrnKdnUzWt4C3k2Iwmpl7iu0zhFB-y1vgwkRKr2XscwG8xe3xSkYdlapBpM9HjsDlRivy1Bz39cOsP5X5F03iqd5jw1EBVKSqFl0dDXEfLeKTjWHLWIHC0_ZqUsycu_OLSLCG2BJSUTyX0KuCYcoa2i-qejjO6p84AZDlUe0jK3hSNVUgz9c7z0uMXQkk5Qn0r7d9fgB-d4CL831g"
DROPBOX_FOLDER = "/tes-ambilfoto"

# ==================== BIZNET NOS CONFIGURATION ====================
NOS_ENDPOINT    = os.getenv('NOS_ENDPOINT',    'https://nos.jkt-1.neo.id')
NOS_ACCESS_KEY  = os.getenv('NOS_ACCESS_KEY',  '1e5546a0c06c65b46766')
NOS_SECRET_KEY  = os.getenv('NOS_SECRET_KEY',  'mgKYlUsVYm5fLpjdgv2PShY7ehr/Giz+X1xcfjXK')
NOS_BUCKET      = os.getenv('NOS_BUCKET',      'master')           # Bucket di Biznet NOS
NOS_FOLDER      = os.getenv('NOS_FOLDER',      'photos')           # Subfolder di dalam bucket

# ==================== STORAGE PATHS ====================
UPLOAD_COMPRESSED = 'uploads/compressed'
UPLOAD_PREVIEW_WATERMARKED = 'uploads/preview_watermarked'
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
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            watermark_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            
            logo = self.watermark_logo.copy()
            logo.thumbnail((logo_size, logo_size), Image.LANCZOS)
            
            if logo.mode == 'RGBA':
                r, g, b, a = logo.split()
                rgb = Image.merge('RGB', (r, g, b))
                enhancer = ImageEnhance.Color(rgb)
                rgb = enhancer.enhance(2.0)
                enhancer = ImageEnhance.Contrast(rgb)
                rgb = enhancer.enhance(1.5)
                r, g, b = rgb.split()
                a = a.point(lambda p: int(p * opacity))
                logo = Image.merge('RGBA', (r, g, b, a))
            
            img_width, img_height = img.size
            logo_width, logo_height = logo.size
            
            watermark_count = 0
            
            for y in range(-logo_height//2, img_height + logo_height//2, spacing):
                for x in range(-logo_width//2, img_width + logo_width//2, spacing):
                    offset_x = (x + y) % 40 - 20
                    offset_y = (y + x) % 40 - 20
                    pos_x = x + offset_x
                    pos_y = y + offset_y
                    try:
                        watermark_layer.paste(logo, (pos_x, pos_y), logo)
                        watermark_count += 1
                    except:
                        pass
            
            print(f"   🔒 Added {watermark_count} VISIBLE watermarks (size: {logo_size}px, opacity: {int(opacity*100)}%)")
            
            watermarked = Image.alpha_composite(img, watermark_layer)
            watermarked = watermarked.convert('RGB')
            return watermarked
            
        except Exception as e:
            print(f"⚠️  Watermark error: {e}")
            import traceback
            traceback.print_exc()
            return img.convert('RGB') if img.mode != 'RGB' else img
    
    def create_preview_version(self, img, max_dimension=800, target_size_kb=150, quality_start=35):
        """✅ OLD METHOD - kept for backward compatibility"""
        return self.create_preview_with_watermark(img, max_dimension, target_size_kb, quality_start)
    
    def create_preview_with_watermark(self, img, max_dimension=1280, target_size_kb=150, quality_start=40):
        """✅ Create LOCAL preview with VISIBLE watermark"""
        try:
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            print(f"   🔒 Adding VISIBLE watermarks to LOCAL preview...")
            img = self.add_multi_watermark(img, logo_size=180, opacity=0.65, spacing=200)
            
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
        """✅ Create compressed version WITHOUT watermark for Dropbox / NOS"""
        try:
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_size = max(img.size)
            if original_size > max_dimension:
                ratio = max_dimension / original_size
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
                print(f"   📐 Resized to: {img.size[0]}x{img.size[1]}")
            
            img = self.apply_smart_enhancement(img)
            
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
            print(f"   💾 Compressed version (NO watermark): {final_size:.1f}KB")
            output.seek(0)
            return output
            
        except Exception as e:
            print(f"❌ Compression error: {e}")
            return None
    
    def create_optimized_original(self, img, max_dimension=2560, target_quality=85):
        """✅ OLD METHOD - kept for backward compatibility"""
        try:
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_size = max(img.size)
            if original_size > max_dimension:
                ratio = max_dimension / original_size
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
                print(f"📐 Resized: {img.size[0]}x{img.size[1]}")
            
            img = self.apply_smart_enhancement(img)
            
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
        """✅ Alias for backward compatibility"""
        return self.create_compressed_for_dropbox(img, max_dimension, target_quality * 7, target_quality)
    
    def apply_smart_enhancement(self, img):
        """Apply subtle enhancement"""
        try:
            img_array = np.array(img)
            kernel = np.array([[-0.5, -0.5, -0.5],
                             [-0.5,  5.0, -0.5],
                             [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(img_array, -1, kernel * 0.1)
            enhanced = cv2.addWeighted(img_array, 0.9, sharpened, 0.1, 0)
            return Image.fromarray(enhanced)
        except Exception as e:
            print(f"⚠️  Enhancement skipped: {e}")
            return img


# ==================== BIZNET NOS MANAGER ====================

class BiznetNOSManager:
    """Manager untuk operasi Biznet NOS (S3-compatible)"""
    
    def __init__(self, endpoint, access_key, secret_key, bucket, folder='photos'):
        self.bucket  = bucket
        self.folder  = folder.strip('/')
        self.s3      = None
        
        try:
            self.s3 = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}   # NOS pakai path-style
                ),
                region_name='jkt-1'
            )
            # Quick connectivity check
            self.s3.head_bucket(Bucket=self.bucket)
            print(f"✅ Biznet NOS connected! Bucket: {self.bucket}")
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == '404':
                print(f"⚠️  NOS: Bucket '{self.bucket}' tidak ditemukan. Buat bucket dulu di dashboard.")
            elif code in ('403', 'AccessDenied'):
                print(f"⚠️  NOS: Access denied ke bucket '{self.bucket}'. Cek permission.")
            else:
                print(f"⚠️  NOS ClientError: {e}")
            self.s3 = None
        except Exception as e:
            print(f"❌ Biznet NOS connection error: {e}")
            self.s3 = None
    
    def _make_key(self, filename):
        """Buat object key dengan subfolder"""
        if self.folder:
            return f"{self.folder}/{filename}"
        return filename
    
    def upload_from_memory(self, file_bytes: bytes, filename: str,
                           content_type: str = 'image/jpeg',
                           public: bool = True) -> dict:
        """
        Upload bytes ke NOS.
        
        Returns:
            {
                'success': bool,
                'key': str,
                'url': str,        # public URL (jika public=True)
                'error': str       # hanya jika gagal
            }
        """
        if not self.s3:
            return {'success': False, 'error': 'NOS not connected'}
        
        key = self._make_key(filename)
        
        extra_args = {'ContentType': content_type}
        if public:
            extra_args['ACL'] = 'public-read'
        
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file_bytes,
                **extra_args
            )
            
            # Build public URL
            endpoint_clean = NOS_ENDPOINT.rstrip('/')
            url = f"{endpoint_clean}/{self.bucket}/{key}"
            
            print(f"   ✅ NOS upload OK: {key} ({len(file_bytes)/1024:.1f}KB)")
            return {'success': True, 'key': key, 'url': url}
            
        except ClientError as e:
            msg = str(e)
            print(f"❌ NOS upload error: {msg}")
            return {'success': False, 'error': msg}
        except Exception as e:
            print(f"❌ NOS unexpected error: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_file(self, key: str) -> BytesIO | None:
        """Download object dari NOS ke BytesIO"""
        if not self.s3:
            return None
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return BytesIO(response['Body'].read())
        except Exception as e:
            print(f"❌ NOS download error: {e}")
            return None
    
    def get_presigned_url(self, key: str, expiry: int = 3600) -> str | None:
        """Generate presigned URL (untuk file private)"""
        if not self.s3:
            return None
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=expiry
            )
            return url
        except Exception as e:
            print(f"❌ NOS presigned URL error: {e}")
            return None
    
    def delete_file(self, key: str) -> bool:
        """Hapus object dari NOS"""
        if not self.s3:
            return False
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            print(f"🗑️  NOS deleted: {key}")
            return True
        except Exception as e:
            print(f"❌ NOS delete error: {e}")
            return False


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
            print(f"❌ Dropbox upload error: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_file(self, dropbox_path):
        if not self.dbx:
            return None
        try:
            metadata, response = self.dbx.files_download(dropbox_path)
            return BytesIO(response.content)
        except Exception as e:
            print(f"❌ Dropbox download error: {e}")
            return None
    
    def get_temporary_link(self, dropbox_path):
        if not self.dbx:
            return None
        try:
            result = self.dbx.files_get_temporary_link(dropbox_path)
            return result.link
        except Exception as e:
            print(f"❌ Dropbox temp link error: {e}")
            return None
    
    def create_folder(self, folder_path):
        if not self.dbx:
            return False
        try:
            self.dbx.files_create_folder_v2(folder_path)
            print(f"📁 Created Dropbox folder: {folder_path}")
            return True
        except ApiError as e:
            if 'conflict' in str(e).lower():
                return True
            else:
                print(f"❌ Dropbox create folder error: {e}")
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
        with torch.no_grad():
            face_img = face_img.to(self.device)
            embedding = self.resnet(face_img.unsqueeze(0))
        return embedding.cpu().numpy().flatten()
    
    def detect_faces_from_image_array(self, img):
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
                    w = x2 - x1
                    h = y2 - y1
                    expand_x = int(w * 0.2)
                    expand_y = int(h * 0.25)
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
        with open('soccer_clinic_db.pkl', 'wb') as f:
            pickle.dump(self.photo_database, f)
    
    def load_database(self):
        if os.path.exists('soccer_clinic_db.pkl'):
            with open('soccer_clinic_db.pkl', 'rb') as f:
                self.photo_database = pickle.load(f)
            print(f"✅ Database loaded: {len(self.photo_database['photos'])} photos")
        else:
            print("📂 New database created")


# ==================== INITIALIZE ALL SYSTEMS ====================

face_system      = SoccerClinicFaceRecognition()
image_processor  = SmartImageProcessor(WATERMARK_LOGO)
dropbox_manager  = DropboxManager(DROPBOX_ACCESS_TOKEN)
nos_manager      = BiznetNOSManager(NOS_ENDPOINT, NOS_ACCESS_KEY, NOS_SECRET_KEY, NOS_BUCKET, NOS_FOLDER)

# Create local directories
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


# ==================== PHOTOGRAPHER UPLOAD ====================

@app.route('/api/photographer/upload', methods=['POST', 'OPTIONS'])
@cross_origin()
def photographer_upload():
    """
    ✅ MODIFIED: Upload photo dengan triple strategy
    1. Local preview WITH watermark (untuk browsing)
    2. Dropbox compressed WITHOUT watermark (untuk purchase - backup)
    3. Biznet NOS compressed WITHOUT watermark (untuk purchase - primary)
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
        filename   = data['filename']
        metadata   = data.get('metadata', {})
        
        complete_metadata = {
            'event_id':          metadata.get('event_id', ''),
            'event_name':        metadata.get('event_name', 'Untitled Event'),
            'event_type':        metadata.get('event_type', 'general'),
            'event_date':        metadata.get('event_date', datetime.now().strftime('%Y-%m-%d')),
            'location':          metadata.get('location', 'Not specified'),
            'photographer_id':   metadata.get('photographer_id', ''),
            'photographer_name': metadata.get('photographer_name', 'Photographer'),
            'uploaded_by':       metadata.get('uploaded_by', ''),
            'upload_timestamp':  metadata.get('upload_timestamp', datetime.now().isoformat())
        }
        
        print('='*70)
        print('📝 TRIPLE STRATEGY UPLOAD')
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
        
        # Unique filename
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
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # ── STEP 1: LOCAL preview WITH watermark ─────────────────────────────
        print('🔒 Creating LOCAL preview WITH watermark...')
        preview_io = image_processor.create_preview_with_watermark(
            img_pil.copy(), max_dimension=1280, target_size_kb=150
        )
        
        preview_path    = os.path.join(UPLOAD_PREVIEW_WATERMARKED, unique_filename)
        preview_success = False
        
        if preview_io:
            with open(preview_path, 'wb') as f:
                f.write(preview_io.getvalue())
            preview_size = os.path.getsize(preview_path) / 1024
            print(f'   ✅ LOCAL preview saved: {preview_size:.1f}KB (WITH watermark)')
            preview_success = True
        
        # ── STEP 2: Compressed WITHOUT watermark (shared for Dropbox & NOS) ──
        print('🗜️  Creating compressed version WITHOUT watermark...')
        compressed_io = image_processor.create_compressed_for_dropbox(
            img_pil.copy(), max_dimension=1920, target_size_kb=500
        )
        compressed_bytes = compressed_io.getvalue() if compressed_io else None
        
        # ── STEP 3: Upload to Biznet NOS ──────────────────────────────────────
        nos_success = False
        nos_key     = None
        nos_url     = None
        
        if compressed_bytes and nos_manager.s3:
            print('🌩️  Uploading to Biznet NOS (NO watermark)...')
            nos_result = nos_manager.upload_from_memory(
                compressed_bytes, unique_filename,
                content_type='image/jpeg', public=True
            )
            if nos_result['success']:
                nos_key     = nos_result['key']
                nos_url     = nos_result['url']
                nos_success = True
                print(f'   ✅ NOS upload OK → {nos_url}')
            else:
                print(f'   ❌ NOS upload failed: {nos_result.get("error")}')
        else:
            if not nos_manager.s3:
                print('   ⚠️  NOS not connected, skipping NOS upload')
        
        # ── STEP 4: Upload to Dropbox (backup) ───────────────────────────────
        dropbox_success = False
        dropbox_path    = None
        dropbox_link    = None
        
        if compressed_bytes and dropbox_manager.dbx:
            print('☁️  Uploading to Dropbox (NO watermark, backup)...')
            dropbox_path   = f"{DROPBOX_FOLDER}/{unique_filename}"
            dropbox_result = dropbox_manager.upload_from_memory(compressed_bytes, dropbox_path)
            if dropbox_result['success']:
                dropbox_link    = dropbox_result.get('shared_link')
                dropbox_success = True
                print(f'   ✅ Dropbox upload OK')
            else:
                print(f'   ❌ Dropbox upload failed: {dropbox_result.get("error")}')
        
        # Backward-compat: also save preview to old compressed folder
        compressed_path = os.path.join(UPLOAD_COMPRESSED, unique_filename)
        if preview_io:
            preview_io.seek(0)
            with open(compressed_path, 'wb') as f:
                f.write(preview_io.read())
        
        # ── Save to local database ────────────────────────────────────────────
        photo_id = str(uuid.uuid4())
        face_system.photo_database['photos'][photo_id] = {
            'path_original':       None,
            'path_compressed':     compressed_path if preview_success else None,
            'path_preview':        preview_path    if preview_success else None,
            'filename':            unique_filename,
            'faces_data':          faces_data,
            'metadata':            complete_metadata,
            'uploaded_at':         datetime.now().isoformat(),
            # Dropbox
            'dropbox_path':        dropbox_path if dropbox_success else None,
            'dropbox_link':        dropbox_link,
            # Biznet NOS
            'nos_key':             nos_key,
            'nos_url':             nos_url,
            # Flags
            'has_watermark':             image_processor.watermark_logo is not None,
            'has_watermark_preview':     preview_success,
            'storage_strategy':          'triple'
        }
        
        face_system.save_database()
        
        print('='*70)
        print('✅ TRIPLE STRATEGY UPLOAD COMPLETED')
        print(f'📌 Photo ID:                  {photo_id}')
        print(f'👤 Faces:                     {len(faces_data)}')
        print(f'🔒 LOCAL Preview (watermark): {"✓" if preview_success else "✗"}')
        print(f'🌩️  Biznet NOS (no watermark): {"✓" if nos_success else "✗"}')
        print(f'☁️  Dropbox (no watermark):    {"✓" if dropbox_success else "✗"}')
        print('='*70 + '\n')
        
        return jsonify({
            'success':           True,
            'photo_id':          photo_id,
            'faces_detected':    len(faces_data),
            'watermarked':       image_processor.watermark_logo is not None,
            'preview_created':   preview_success,
            'nos_uploaded':      nos_success,
            'nos_url':           nos_url,
            'dropbox_uploaded':  dropbox_success,
            'compressed':        preview_success,
            'storage_strategy':  'triple'
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
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photos = []
        for photo_id, photo_data in face_system.photo_database['photos'].items():
            photos.append({
                'photo_id':               photo_id,
                'filename':               photo_data['filename'],
                'faces_count':            len(photo_data.get('faces_data', [])),
                'metadata':               photo_data.get('metadata', {}),
                'uploaded_at':            photo_data.get('uploaded_at', ''),
                'in_dropbox':             photo_data.get('dropbox_path') is not None,
                'in_nos':                 photo_data.get('nos_key') is not None,
                'nos_url':                photo_data.get('nos_url'),
                'has_visible_watermark':  photo_data.get('has_watermark', False),
                'has_watermark_preview':  photo_data.get('has_watermark_preview', False),
                'storage_strategy':       photo_data.get('storage_strategy', 'unknown')
            })
        
        photos.sort(key=lambda x: x['uploaded_at'], reverse=True)
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []})


@app.route('/api/user/register_face', methods=['POST', 'OPTIONS'])
@cross_origin()
def user_register_face():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        image_data = data['image']
        
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_data = image_data.strip()
            image_bytes = base64.b64decode(image_data)
            nparr  = np.frombuffer(image_bytes, np.uint8)
            frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'success': False, 'error': 'Failed to decode image - invalid image data'}), 400
        except Exception as decode_error:
            return jsonify({'success': False, 'error': f'Image decode error: {str(decode_error)}'}), 400
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        print(f"🔍 Detecting faces in image {img_pil.size}...")
        boxes, probs = face_system.mtcnn.detect(img_pil)
        
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'error': 'No face detected in image', 'faces_detected': 0}), 200
        
        try:
            box = boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return jsonify({'success': False, 'error': 'Invalid face region detected'}), 200
            
            face_img = img_pil.crop((x1, y1, x2, y2))
            face_img = face_img.resize((160, 160), Image.LANCZOS)
            
            face_array  = np.array(face_img).astype(np.float32)
            face_array  = (face_array - 127.5) / 128.0
            face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
            
            embedding = face_system.extract_embedding(face_tensor)
            
            if embedding is None or len(embedding) == 0:
                return jsonify({'success': False, 'error': 'Failed to extract face embedding'}), 200
            
            return jsonify({
                'success':        True,
                'embedding':      embedding.tolist(),
                'faces_detected': len(boxes),
                'confidence':     float(probs[0])
            }), 200
            
        except Exception as extraction_error:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Face extraction failed: {str(extraction_error)}'}), 500
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user/my_photos', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_user_photos():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data           = request.json
        user_embedding = data.get('embedding')
        user_id        = data.get('user_id')
        
        if not user_embedding:
            return jsonify({'success': False, 'error': 'Embedding required'}), 400
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID required'}), 400
        
        print(f"🔍 Face matching for user: {user_id}")
        matched_photos = face_system.match_user_face(user_embedding)
        
        saved_count = 0
        for match in matched_photos:
            event_photo_id = get_event_photo_id_by_ai_photo_id(match['photo_id'])
            if event_photo_id:
                if save_photo_match(user_id, event_photo_id, match['distance'],
                                    1 - match['distance'], match['cosine_similarity']):
                    saved_count += 1
        
        photos = []
        for match in matched_photos:
            photo_data = match['photo_data']
            photos.append({
                'photo_id':              match['photo_id'],
                'filename':              photo_data['filename'],
                'metadata':              photo_data.get('metadata', {}),
                'distance':              match['distance'],
                'cosine_similarity':     match['cosine_similarity'],
                'in_dropbox':            photo_data.get('dropbox_path') is not None,
                'in_nos':                photo_data.get('nos_key') is not None,
                'nos_url':               photo_data.get('nos_url'),
                'has_visible_watermark': photo_data.get('has_watermark', False),
                'storage_strategy':      photo_data.get('storage_strategy', 'unknown')
            })
        
        return jsonify({'success': True, 'photos': photos, 'matches_saved_to_db': saved_count})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []}), 500


# ==================== IMAGE SERVING & DOWNLOAD ====================

@app.route('/api/image/preview/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def serve_preview_image(photo_id):
    """Serve WATERMARKED preview (local file)"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'error': 'Photo not found'}), 404
        
        for path_key in ('path_preview', 'path_compressed'):
            path = photo_data.get(path_key)
            if path and os.path.exists(path):
                return send_file(path, mimetype='image/jpeg', as_attachment=False)
        
        return jsonify({'error': 'Preview not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/nos/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_from_nos(photo_id):
    """
    🆕 Download compressed photo WITHOUT watermark dari Biznet NOS.
    Primary download endpoint.
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        nos_url = photo_data.get('nos_url')
        nos_key = photo_data.get('nos_key')
        
        if not nos_key:
            # Fallback ke Dropbox kalau tidak ada di NOS
            return redirect(url_for('download_from_dropbox', photo_id=photo_id))
        
        if not nos_manager.s3:
            return jsonify({'success': False, 'error': 'NOS not available'}), 503
        
        # Jika file public, redirect langsung ke URL
        if nos_url:
            return redirect(nos_url)
        
        # Fallback: download via presigned URL
        presigned = nos_manager.get_presigned_url(nos_key)
        if presigned:
            return redirect(presigned)
        
        # Last resort: stream file
        file_stream = nos_manager.download_file(nos_key)
        if file_stream:
            filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
            return send_file(file_stream, mimetype='image/jpeg',
                             as_attachment=True, download_name=filename)
        
        return jsonify({'success': False, 'error': 'Failed to download from NOS'}), 500
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download/dropbox/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_from_dropbox(photo_id):
    """Download compressed photo WITHOUT watermark dari Dropbox (backup)"""
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
        except:
            pass
        
        file_stream = dropbox_manager.download_file(dropbox_path)
        if file_stream:
            filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
            return send_file(file_stream, mimetype='image/jpeg',
                             as_attachment=True, download_name=filename)
        
        return jsonify({'success': False, 'error': 'Failed to download from Dropbox'}), 500
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photo_id': photo_id}), 500


@app.route('/api/download/purchased/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_purchased_photo(photo_id):
    """
    ✅ Unified endpoint untuk download foto yang sudah dibeli.
    Priority: NOS → Dropbox
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        # 1. Coba NOS dulu (primary)
        nos_key = photo_data.get('nos_key')
        if nos_key and nos_manager.s3:
            nos_url = photo_data.get('nos_url')
            if nos_url:
                print(f'✅ Redirect ke NOS URL: {nos_url}')
                return redirect(nos_url)
            
            presigned = nos_manager.get_presigned_url(nos_key)
            if presigned:
                return redirect(presigned)
            
            file_stream = nos_manager.download_file(nos_key)
            if file_stream:
                filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
                return send_file(file_stream, mimetype='image/jpeg',
                                 as_attachment=True, download_name=filename)
        
        # 2. Fallback ke Dropbox
        dropbox_path = photo_data.get('dropbox_path')
        if dropbox_path and dropbox_manager.dbx:
            try:
                metadata, response = dropbox_manager.dbx.files_download(dropbox_path)
                filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
                return send_file(BytesIO(response.content), mimetype='image/jpeg',
                                 as_attachment=True, download_name=filename)
            except:
                pass
        
        return jsonify({'success': False, 'error': 'File not available in any storage'}), 404
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/photo/download_direct/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_photo_direct(photo_id):
    """✅ KEPT: Download photo directly as binary (backward compat)"""
    if request.method == 'OPTIONS':
        return '', 204
    return redirect(url_for('download_purchased_photo', photo_id=photo_id))


@app.route('/api/photo/get_base64/<photo_id>', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_photo_base64(photo_id):
    """✅ KEPT: Get photo as base64 JSON"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        file_bytes = None
        
        # Try NOS first
        nos_key = photo_data.get('nos_key')
        if nos_key and nos_manager.s3:
            stream = nos_manager.download_file(nos_key)
            if stream:
                file_bytes = stream.read()
        
        # Fallback Dropbox
        if not file_bytes:
            dropbox_path = photo_data.get('dropbox_path')
            if dropbox_path and dropbox_manager.dbx:
                _, response = dropbox_manager.dbx.files_download(dropbox_path)
                file_bytes = response.content
        
        if not file_bytes:
            return jsonify({'success': False, 'error': 'File not available'}), 404
        
        base64_data = base64.b64encode(file_bytes).decode('utf-8')
        
        return jsonify({
            'success':    True,
            'photo_id':   photo_id,
            'filename':   photo_data.get('filename'),
            'face_image': f'data:image/jpeg;base64,{base64_data}',
            'metadata':   photo_data.get('metadata', {})
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/photo/generate_optimized/<photo_id>', methods=['POST', 'OPTIONS'])
@cross_origin()
def generate_optimized_version(photo_id):
    """✅ KEPT: Generate optimized version (backward compat)"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        if photo_data.get('optimized_path') or photo_data.get('nos_key'):
            return jsonify({
                'success':        True,
                'optimized_path': photo_data.get('optimized_path') or photo_data.get('nos_key'),
                'nos_url':        photo_data.get('nos_url'),
                'already_exists': True
            })
        
        # Download source
        file_bytes = None
        dropbox_path = photo_data.get('dropbox_path')
        if dropbox_path and dropbox_manager.dbx:
            _, response = dropbox_manager.dbx.files_download(dropbox_path)
            file_bytes = response.content
        
        if not file_bytes:
            return jsonify({'success': False, 'error': 'Source file not available'}), 404
        
        # Create optimized
        nparr   = np.frombuffer(file_bytes, np.uint8)
        img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        opt_io = image_processor.create_compressed_for_dropbox(img_pil, max_dimension=1920, target_size_kb=500)
        if not opt_io:
            return jsonify({'success': False, 'error': 'Compression failed'}), 500
        
        opt_bytes = opt_io.getvalue()
        filename  = photo_data.get('filename')
        
        # Upload to NOS
        nos_result = nos_manager.upload_from_memory(opt_bytes, f"optimized_{filename}")
        if nos_result['success']:
            photo_data['nos_key'] = nos_result['key']
            photo_data['nos_url'] = nos_result['url']
            face_system.save_database()
            return jsonify({
                'success':   True,
                'nos_key':   nos_result['key'],
                'nos_url':   nos_result['url'],
                'size_kb':   len(opt_bytes) / 1024
            })
        
        return jsonify({'success': False, 'error': 'Upload failed'}), 500
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/photo/download_optimized/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_optimized(photo_id):
    """✅ KEPT: Download optimized (backward compat) → unified purchased endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    return redirect(url_for('download_purchased_photo', photo_id=photo_id))


@app.route('/api/photographer/delete/<photo_id>', methods=['DELETE', 'OPTIONS'])
@cross_origin()
def delete_photo(photo_id):
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        
        results = {}
        
        # Delete from NOS
        nos_key = photo_data.get('nos_key')
        if nos_key:
            results['nos_deleted'] = nos_manager.delete_file(nos_key)
        
        # Delete from Dropbox
        if photo_data.get('dropbox_path') and dropbox_manager.dbx:
            try:
                dropbox_manager.dbx.files_delete_v2(photo_data['dropbox_path'])
                results['dropbox_deleted'] = True
            except:
                results['dropbox_deleted'] = False
        
        # Delete local files
        for path_key in ('path_preview', 'path_compressed'):
            path = photo_data.get(path_key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    results[f'{path_key}_deleted'] = True
                except:
                    pass
        
        # Delete embeddings
        for face in photo_data.get('faces_data', []):
            eid = face.get('embedding_id')
            if eid in face_system.photo_database['face_embeddings']:
                del face_system.photo_database['face_embeddings'][eid]
        
        del face_system.photo_database['photos'][photo_id]
        face_system.save_database()
        
        return jsonify({'success': True, 'message': 'Photo deleted', **results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({
        'status':        'ok',
        'service':       'AI Face Recognition - Triple Strategy (NOS + Dropbox + Local)',
        'total_photos':  len(face_system.photo_database['photos']),
        'total_faces':   len(face_system.photo_database['face_embeddings']),
        'storage': {
            'nos': {
                'connected': nos_manager.s3 is not None,
                'endpoint':  NOS_ENDPOINT,
                'bucket':    NOS_BUCKET,
                'folder':    NOS_FOLDER,
            },
            'dropbox': {
                'connected': dropbox_manager.dbx is not None,
                'folder':    DROPBOX_FOLDER,
            },
            'local_preview': 'WITH watermark',
        },
        'watermark_available': image_processor.watermark_logo is not None,
        'storage_strategy': {
            'preview':    'LOCAL with VISIBLE watermark',
            'nos':        'COMPRESSED without watermark (primary)',
            'dropbox':    'COMPRESSED without watermark (backup)',
            'hd_files':   'NONE - all compressed'
        },
        'endpoints': {
            'new': [
                '/api/download/nos/<id>       ← NOS (primary)',
                '/api/download/purchased/<id> ← auto NOS→Dropbox fallback',
            ],
            'kept': [
                '/api/download/dropbox/<id>',
                '/api/photo/download_direct/<id>',
                '/api/photo/get_base64/<id>',
                '/api/photo/generate_optimized/<id>',
                '/api/photo/download_optimized/<id>',
            ]
        }
    })


# ==================== BATCH PROCESSING ====================

def batch_process_from_folder(folder_path):
    """Batch process images dengan triple strategy"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    metadata = {
        'event_name':   input("Event Name: ") or 'Batch Upload',
        'location':     input("Location: ") or 'Unknown',
        'photographer': input("Photographer: ") or 'Anonymous',
        'date':         datetime.now().isoformat()
    }
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = set()
    for ext in image_extensions:
        image_files.update(glob.glob(os.path.join(folder_path, ext)))
    image_files = sorted(list(image_files))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    total = len(image_files)
    print(f"\n{'='*70}")
    print(f"⚽ TRIPLE STRATEGY BATCH PROCESSING")
    print(f"{'='*70}")
    print(f"📸 Total: {total}")
    print(f"🌩️  NOS:     {NOS_BUCKET}/{NOS_FOLDER} (primary)")
    print(f"☁️  Dropbox: {DROPBOX_FOLDER} (backup)")
    print(f"{'='*70}\n")
    
    if input(f"Process {total} images? (y/n): ").lower() != 'y':
        print("❌ Cancelled")
        return
    
    processed = failed = uploaded_nos = uploaded_dropbox = 0
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            filename = os.path.basename(img_path)
            print(f"[{idx}/{total}] {filename}... ", end='', flush=True)
            
            img = cv2.imread(img_path)
            if img is None:
                print("❌ Failed to read")
                failed += 1
                continue
            
            name, ext = os.path.splitext(filename)
            unique_filename = filename
            counter = 1
            while any(p['filename'] == unique_filename for p in face_system.photo_database['photos'].values()):
                unique_filename = f"{name}_{counter}{ext}"
                counter += 1
            
            faces_data = face_system.detect_faces_from_image_array(img)
            img_pil    = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            preview_io   = image_processor.create_preview_with_watermark(img_pil.copy())
            preview_path = os.path.join(UPLOAD_PREVIEW_WATERMARKED, unique_filename)
            if preview_io:
                with open(preview_path, 'wb') as f:
                    f.write(preview_io.getvalue())
            
            compressed_io    = image_processor.create_compressed_for_dropbox(img_pil.copy())
            compressed_bytes = compressed_io.getvalue() if compressed_io else None
            
            nos_key = nos_url = dropbox_path = dropbox_link = None
            
            if compressed_bytes and nos_manager.s3:
                r = nos_manager.upload_from_memory(compressed_bytes, unique_filename)
                if r['success']:
                    nos_key  = r['key']
                    nos_url  = r['url']
                    uploaded_nos += 1
            
            if compressed_bytes and dropbox_manager.dbx:
                dropbox_path = f"{DROPBOX_FOLDER}/{unique_filename}"
                r = dropbox_manager.upload_from_memory(compressed_bytes, dropbox_path)
                if r['success']:
                    dropbox_link = r.get('shared_link')
                    uploaded_dropbox += 1
            
            photo_id = str(uuid.uuid4())
            face_system.photo_database['photos'][photo_id] = {
                'path_original':         None,
                'path_compressed':       preview_path,
                'path_preview':          preview_path,
                'filename':              unique_filename,
                'faces_data':            faces_data,
                'metadata':              metadata.copy(),
                'uploaded_at':           datetime.now().isoformat(),
                'dropbox_path':          dropbox_path,
                'dropbox_link':          dropbox_link,
                'nos_key':               nos_key,
                'nos_url':               nos_url,
                'has_watermark':         image_processor.watermark_logo is not None,
                'has_watermark_preview': True,
                'storage_strategy':      'triple'
            }
            
            processed += 1
            nos_mark = '🌩️' if nos_key else '✗'
            dbx_mark = '☁️' if dropbox_path else '✗'
            print(f"✅ {len(faces_data)} faces | NOS:{nos_mark} Dropbox:{dbx_mark}")
            
            if processed % 10 == 0:
                face_system.save_database()
                print(f"💾 Checkpoint ({processed}/{total})")
                
        except Exception as e:
            print(f"❌ {e}")
            failed += 1
    
    face_system.save_database()
    
    print(f"\n{'='*70}")
    print(f"✅ BATCH DONE: {processed}/{total} | ❌ Failed: {failed}")
    print(f"🌩️  NOS: {uploaded_nos} | ☁️  Dropbox: {uploaded_dropbox}")
    print(f"{'='*70}\n")


def generate_self_signed_cert():
    try:
        from OpenSSL import crypto
    except ImportError:
        print("⚠️  PyOpenSSL not installed. pip install pyopenssl")
        return False
    
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("✅ SSL Certificate exists")
        return True
    
    try:
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        cert = crypto.X509()
        cert.get_subject().C  = "ID"
        cert.get_subject().ST = "East Kalimantan"
        cert.get_subject().L  = "Samarinda"
        cert.get_subject().O  = "Soccer Clinic"
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
        print(f"❌ Error: {e}")
        return False


# ==================== MAIN ====================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch' and len(sys.argv) > 2:
            batch_process_from_folder(sys.argv[2])
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("""
USAGE:
  python app.py                    # Run web server
  python app.py --batch <folder>   # Batch process
  python app.py --help             # Show help

STORAGE STRATEGY (TRIPLE):
  LOCAL:   WITH watermark, ~150KB
  NOS:     WITHOUT watermark, ~500KB  ← primary download
  Dropbox: WITHOUT watermark, ~500KB  ← backup download

REQUIRED:
  pip install boto3

ENV VARS (optional override):
  NOS_ENDPOINT   = https://nos.jkt-1.neo.id
  NOS_ACCESS_KEY = 1e5546a0c06c65b46766
  NOS_SECRET_KEY = mgKYlUsVYm5fLpjdgv2PShY7ehr/Giz+X1xcfjXK
  NOS_BUCKET     = ambilfoto
  NOS_FOLDER     = photos
""")
            sys.exit(0)
        sys.exit(1)
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*70)
    print("⚽ AI FACE RECOGNITION - TRIPLE STRATEGY")
    print("="*70)
    print(f"🌩️  Biznet NOS:    {'✅ Connected' if nos_manager.s3     else '❌ Not Connected'}")
    print(f"☁️  Dropbox:        {'✅ Connected' if dropbox_manager.dbx else '❌ Not Connected'}")
    print(f"🔒 Watermark:      {'✅ Available' if image_processor.watermark_logo else '⚠️  Not Found'}")
    print(f"\n💾 STORAGE:")
    print(f"   LOCAL   → WITH watermark (~150KB)")
    print(f"   NOS     → NO watermark (~500KB) [primary]")
    print(f"   Dropbox → NO watermark (~500KB) [backup]")
    print("="*70)
    
    use_https = input("\nUse HTTPS? (y/n): ").lower() == 'y'
    
    if use_https and generate_self_signed_cert():
        print(f"\n🌐 HTTPS: https://localhost:4000 | https://{local_ip}:4000")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('cert.pem', 'key.pem')
        app.run(host='0.0.0.0', port=4000, debug=False, threaded=True, ssl_context=context)
    else:
        print(f"\n🌐 HTTP: http://localhost:4000 | http://{local_ip}:4000")
        app.run(host='0.0.0.0', port=4000, debug=False, threaded=True)