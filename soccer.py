from flask import Flask, render_template, request, jsonify, send_file, Response, redirect
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
from PIL import Image
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
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# MySQL Configuration
MYSQL_CONFIG = {
    'host': os.getenv('DB_HOST', '172.28.176.1'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'af_db')
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
        
        # Check if match already exists
        check_query = """
            SELECT id FROM user_photo_matches 
            WHERE user_id = %s AND event_photo_id = %s
        """
        cursor.execute(check_query, (user_id, event_photo_id))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing match
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
            # Insert new match
            import uuid
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
DROPBOX_ACCESS_TOKEN ="sl.u.AGPZ--_8-wCocOoUwuhRL9nnFc68rDYfBI5JrXI9R2ZL5U9ms92GGm9yqg6vEyHRgQ3i2UuUQ8hk2RgxLavTJQxW1hxGWM2QL7GEn5IRMTKYQSQOhE5FXmgjtnO4j_yEodRT3QGoJNF4vEcpAfmVj03fI-jRPp7p-6nVprpuHnjLrsw5lgKqR9kzrFlmhp4eyV1w7zV78dufvZiEN7gyqXcxkYUrgY-nsEtAeDWjMrm8qA85MdAnUjFukRgA_ZzfEVCCH51lP2a7IHR82aznnmhveXzhOWzxDBhuOHXPCUm86wWqZpfJxxuKmC824WtiJ329P8EzD3wP_OzajI30HW3_axhxuN5I7uvSNNrDWmA6M_8hJANbxz2dQff7rpRSc1Da7I7W1eukPToKqrDrv60GpXnHJsYJCXP0cnIPLdzPUagyH6-MMVV0Jt6rNccCe-fLeb8SBrNZlNcB2tyOaQ_RQLxtSMaqPLvUGzBecFJDh0HqtPDIxBgouQZynuvXi45px_SuM6J90Z9MGrHrlKK5F0jvWYn6PShVnDLdE3gbCY4ApcsuuqqPbsC1gWVEc643gMvBn8hUoUO51uLdhWjp4sxWTEhzOPMmdyRyk2Ss4z_gdwkXi-CDWhOINZ0WfVfdWyJU--X_qhbFbtf2IKARBAz5-SOku3AmOpxbL7U32fFC9ZlmuwFpewRR_3dQz7U4Wlt-oz00CRQZP_E3Sawco1QsYnEUUbP2yW8tfw3695lyRAkJWwy0dfpA9HODk0AjKGaTuLXo1k04Mx0by06mCWzO2L2axR5XO2dK5CRA9FkIuNvwVpmusjPA-kYO76z7uJhtR9XIZgWrdTTfl_zGQvZDkAHHtIaeUQaEJwVCWFrMViwszR8cs1Z52rs5FS2aP8XuGeJjvk4gt7JIDH00lLvI-rYS2xAlDJHpbenbRzY047BTYCo4h6DK1QHDlidmAxaRVQgZgI-w6zW8gxZ86FEOEKObOo0Q-d7_GavNce38Whekbq19yq5CZiu6hx1Z5bcLXyFfAOkLtlepEZJDA2eTaXEtp_pdOltT6ZXOU92QJG7nqaJqfI9KFMbICShudkypK-PFPCVsaAKiMTm4k2ObGdrcpqh75suI21VHLOxh2nDjN_rfkdiL5Jb8gLmlqNGG0sP0MJBIrBwT4MUKDTWk2eDDvP0fCVc564Urs0ZEZU9sE260mFe-5cfwQkTtbQ_doypXoK68lusCmhaXCEThgXfVGnPeg22vQ-BTHa7_ouhY_ONMRfkKYm2SzKNHs8UcrgZM18SfHVW0ocajwqGRqA78vwf7bMuWo3NindZmAs6_oJsRXIh98huBWs4VgBE4YtP7WHp_6v9Ys7dq5A_bChCN_j2AcuM3c2cnhOG35L6mEVQgZC3YO9Dq7YtE33ziemJ4cYuZ5Pz7v7sW"
DROPBOX_FOLDER = "/tes-ambilfoto"

# ==================== STORAGE PATHS ====================
UPLOAD_COMPRESSED = 'uploads/compressed' # Compressed files (untuk preview)
UPLOAD_TEMP = 'uploads/temp'             # Temporary processing

class DropboxManager:
    """Manager untuk semua operasi Dropbox"""
    
    def __init__(self, access_token):
        """Initialize Dropbox client"""
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
        """Upload file directly from memory to Dropbox"""
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
            
            print(f"✅ Uploaded to Dropbox: {dropbox_path}")
            return {
                'success': True,
                'path': dropbox_path,
                'shared_link': download_link
            }
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_file(self, dropbox_path):
        """Download file from Dropbox with better error handling"""
        if not self.dbx:
            print("❌ Dropbox not initialized")
            return None
        
        try:
            print(f"   Downloading from Dropbox: {dropbox_path}")
            metadata, response = self.dbx.files_download(dropbox_path)
            print(f"   ✅ Downloaded {len(response.content)} bytes")
            return BytesIO(response.content)
            
        except ApiError as e:
            print(f"   ❌ Dropbox API Error: {str(e)}")
            return None
            
        except Exception as e:
            print(f"   ❌ Download error: {str(e)}")
            return None
    
    def get_temporary_link(self, dropbox_path):
        """Get temporary download link (4 hour expiry) with better error handling"""
        if not self.dbx:
            print("❌ Dropbox not initialized")
            return None
        
        try:
            print(f"   Requesting temp link for: {dropbox_path}")
            result = self.dbx.files_get_temporary_link(dropbox_path)
            print(f"   ✅ Temp link generated: {result.link[:50]}...")
            return result.link
            
        except ApiError as e:
            error_str = str(e)
            print(f"   ❌ Dropbox API Error: {error_str}")
            
            # Check for specific errors
            if 'not_found' in error_str.lower():
                print(f"   📌 File not found in Dropbox: {dropbox_path}")
            elif 'path' in error_str.lower():
                print(f"   📌 Invalid path: {dropbox_path}")
            elif 'auth' in error_str.lower():
                print(f"   📌 Authentication error - token may be expired")
            
            return None
            
        except Exception as e:
            print(f"   ❌ Unexpected error: {str(e)}")
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
            error_string = str(e)
            if 'conflict' in error_string.lower() or 'folder' in error_string.lower():
                print(f"✅ Folder already exists: {folder_path}")
                return True
            else:
                print(f"❌ Create folder error: {e}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error creating folder: {e}")
            return False

dropbox_manager = DropboxManager(DROPBOX_ACCESS_TOKEN)

if dropbox_manager.dbx:
    try:
        dropbox_manager.create_folder(DROPBOX_FOLDER)
    except Exception as e:
        print(f"⚠️  Warning: Could not create Dropbox folder: {e}")


class ImageCompressor:
    """Compress images for fast preview"""
    
    @staticmethod
    def compress_for_preview(image_path, output_path, max_size_kb=100, quality_start=25):
        """Compress image for preview with aggressive settings"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for preview (max 1280px)
            max_dimension = 1280
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            # Compress to target size
            quality = quality_start
            while quality >= 10:
                img.save(output_path, format='JPEG', quality=quality, optimize=True, progressive=True)
                
                size_kb = os.path.getsize(output_path) / 1024
                if size_kb <= max_size_kb:
                    break
                quality -= 5
            
            return True
            
        except Exception as e:
            print(f"Error compressing image: {e}")
            return False
    
    @staticmethod
    def compress_to_memory(image_path, max_size_kb=100, quality_start=25):
        """Compress image to BytesIO for serving"""
        try:
            img = Image.open(image_path)
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            max_dimension = 1280
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            output = BytesIO()
            quality = quality_start
            
            while quality >= 10:
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
                
                size_kb = output.tell() / 1024
                if size_kb <= max_size_kb:
                    break
                quality -= 5
            
            output.seek(0)
            return output
            
        except Exception as e:
            print(f"Error compressing image: {e}")
            return None


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
        """Detect faces from numpy array image (NOT from file path)"""
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


face_system = SoccerClinicFaceRecognition()
compressor = ImageCompressor()

# Create necessary directories
for directory in [UPLOAD_COMPRESSED, UPLOAD_TEMP]:
    os.makedirs(directory, exist_ok=True)
# HTML Template

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/photographer/upload', methods=['POST', 'OPTIONS'])
@cross_origin()
def photographer_upload():
    """Upload photo directly to Dropbox (NO local original storage)"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # Validate required fields
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        if 'image' not in data or 'filename' not in data:
            return jsonify({
                'success': False,
                'error': 'image and filename are required'
            }), 400
        
        image_data = data['image']
        filename = data['filename']
        metadata = data.get('metadata', {})
        
        # Ensure metadata has all required fields with defaults
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
        print('📝 Upload Request Received')
        print('='*70)
        print(f'📸 Filename: {filename}')
        print(f'📋 Metadata:')
        for key, value in complete_metadata.items():
            print(f'   {key}: {value}')
        print('='*70)
        
        # Decode image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({
                    'success': False,
                    'error': 'Invalid image data'
                }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to decode image: {str(e)}'
            }), 400
        
        # Generate unique filename
        name, ext = os.path.splitext(filename)
        unique_filename = filename
        counter = 1
        
        while any(p['filename'] == unique_filename for p in face_system.photo_database['photos'].values()):
            unique_filename = f"{name}_{counter}{ext}"
            counter += 1
        
        if unique_filename != filename:
            print(f'🔄 Renamed: {filename} → {unique_filename}')
        
        # STEP 1: Face detection
        print('🔍 Detecting faces...')
        faces_data = face_system.detect_faces_from_image_array(img)
        print(f'✅ Detected {len(faces_data)} face(s)')
        
        # STEP 2: Convert to bytes
        _, buffer = cv2.imencode(ext if ext else '.jpg', img)
        image_bytes = buffer.tobytes()
        
        # STEP 3: Upload to Dropbox
        print('☁️  Uploading to Dropbox...')
        dropbox_path = f"{DROPBOX_FOLDER}/{unique_filename}"
        dropbox_result = dropbox_manager.upload_from_memory(image_bytes, dropbox_path)
        
        if not dropbox_result['success']:
            error_msg = dropbox_result.get('error', 'Unknown error')
            print(f'❌ Dropbox upload failed: {error_msg}')
            return jsonify({
                'success': False,
                'error': f"Dropbox upload failed: {error_msg}"
            }), 500
        
        print(f'✅ Uploaded to: {dropbox_path}')
        
        # STEP 4: Create compressed version
        print('📦 Creating compressed version...')
        compressed_path = os.path.join(UPLOAD_COMPRESSED, unique_filename)
        temp_path = os.path.join(UPLOAD_TEMP, unique_filename)
        
        try:
            cv2.imwrite(temp_path, img)
            compress_success = compressor.compress_for_preview(
                temp_path,
                compressed_path,
                max_size_kb=100
            )
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if compress_success:
                compressed_size = os.path.getsize(compressed_path) / 1024
                print(f'✅ Compressed: {compressed_size:.1f}KB')
            else:
                print('⚠️  Compression failed')
                compressed_path = None
        except Exception as e:
            print(f'⚠️  Compression error: {str(e)}')
            compressed_path = None
        
        # STEP 5: Save to database
        photo_id = str(uuid.uuid4())
        face_system.photo_database['photos'][photo_id] = {
            'path_original': None,
            'path_compressed': compressed_path,
            'filename': unique_filename,
            'faces_data': faces_data,
            'metadata': complete_metadata,
            'uploaded_at': datetime.now().isoformat(),
            'dropbox_path': dropbox_path,
            'dropbox_link': dropbox_result.get('shared_link', None)
        }
        
        face_system.save_database()
        
        print('='*70)
        print('✅ UPLOAD COMPLETED')
        print(f'📌 Photo ID: {photo_id}')
        print(f'👤 Faces: {len(faces_data)}')
        print(f'☁️  Dropbox: ✓')
        print(f'📦 Compressed: {"✓" if compress_success else "✗"}')
        print('='*70 + '\n')
        
        return jsonify({
            'success': True,
            'photo_id': photo_id,
            'faces_detected': len(faces_data),
            'dropbox_uploaded': True,
            'compressed': compress_success
        }), 201
        
    except Exception as e:
        print('='*70)
        print('❌ UPLOAD ERROR')
        print(f'Error: {str(e)}')
        print('='*70 + '\n')
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@app.route('/api/photographer/photos', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_photographer_photos():
    """Get all photographer photos"""
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
                'in_dropbox': photo_data.get('dropbox_path') is not None
            })
        
        photos.sort(key=lambda x: x['uploaded_at'], reverse=True)
        
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []})

@app.route('/api/user/register_face', methods=['POST', 'OPTIONS'])
@cross_origin()
def user_register_face():
    """User registers their face"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        image_data = data['image']
        
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        boxes, _ = face_system.mtcnn.detect(img_pil)
        
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'error': 'No face detected'})
        
        box = boxes[0]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        face_img = img_pil.crop((x1, y1, x2, y2))
        face_img = face_img.resize((160, 160))
        
        face_array = np.array(face_img).astype(np.float32)
        face_array = (face_array - 127.5) / 128.0
        face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
        
        embedding = face_system.extract_embedding(face_tensor)
        
        return jsonify({
            'success': True,
            'embedding': embedding.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/user/my_photos', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_user_photos():
    """Get user's matched photos - WITH DATABASE SYNC"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # ✅ DEBUG: Log the raw request
        print("="*70)
        print("📥 REQUEST DEBUG INFO")
        print("="*70)
        print(f"Content-Type: {request.content_type}")
        print(f"Request data type: {type(request.data)}")
        print(f"Request data: {request.data[:200] if request.data else 'None'}")
        
        data = request.json
        print(f"Parsed JSON: {data is not None}")
        
        if data:
            print(f"Keys in data: {list(data.keys())}")
            print(f"user_id value: {data.get('user_id')}")
            print(f"user_id type: {type(data.get('user_id'))}")
            print(f"embedding exists: {data.get('embedding') is not None}")
            print(f"embedding length: {len(data.get('embedding', [])) if isinstance(data.get('embedding'), list) else 'N/A'}")
        
        print("="*70)
        
        user_embedding = data.get('embedding')
        user_id = data.get('user_id')
        
        # ✅ Better error messages
        if not user_embedding:
            return jsonify({
                'success': False, 
                'error': 'Embedding is required',
                'debug': {'received_keys': list(data.keys()) if data else []}
            }), 400
        
        if not user_id:
            return jsonify({
                'success': False, 
                'error': 'User ID is required',
                'debug': {
                    'received_keys': list(data.keys()) if data else [],
                    'user_id_value': user_id,
                    'user_id_type': str(type(user_id))
                }
            }), 400
        
        print(f"🔍 Face matching for user: {user_id}")
        
        matched_photos = face_system.match_user_face(user_embedding)
        
        # Save matches to database
        saved_count = 0
        for match in matched_photos:
            photo_id = match['photo_id']
            distance = match['distance']
            cosine_sim = match['cosine_similarity']
            
            # Get event_photo_id from ai_photo_id
            event_photo_id = get_event_photo_id_by_ai_photo_id(photo_id)
            
            if event_photo_id:
                similarity_score = 1 - distance
                confidence_score = cosine_sim
                
                if save_photo_match(user_id, event_photo_id, distance, similarity_score, confidence_score):
                    saved_count += 1
        
        print(f"✅ Saved {saved_count}/{len(matched_photos)} matches to database")
        
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
                'in_dropbox': photo_data.get('dropbox_path') is not None
            })
        
        return jsonify({
            'success': True, 
            'photos': photos,
            'matches_saved_to_db': saved_count
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e), 
            'photos': []
        }), 500

@app.route('/api/image/preview/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def serve_preview_image(photo_id):
    """Serve COMPRESSED image for fast preview"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            return jsonify({'error': 'Photo not found'}), 404
        
        # Use compressed version for preview
        compressed_path = photo_data.get('path_compressed')
        
        if compressed_path and os.path.exists(compressed_path):
            return send_file(
                compressed_path,
                mimetype='image/jpeg',
                as_attachment=False
            )
        
        # Fallback to original if compressed not available
        original_path = photo_data.get('path_original')
        if original_path and os.path.exists(original_path):
            # Compress on-the-fly
            compressed_io = compressor.compress_to_memory(original_path, max_size_kb=100)
            if compressed_io:
                return send_file(
                    compressed_io,
                    mimetype='image/jpeg',
                    as_attachment=False
                )
            else:
                return send_file(
                    original_path,
                    mimetype='image/jpeg',
                    as_attachment=False
                )
        
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"❌ Preview error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/dropbox/<photo_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_from_dropbox(photo_id):
    """Download ORIGINAL file from Dropbox with better error handling"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        print(f"\n{'='*60}")
        print(f"📥 Download Request for photo_id: {photo_id}")
        
        # Check if photo exists in database
        photo_data = face_system.photo_database['photos'].get(photo_id)
        if not photo_data:
            print(f"❌ Photo not found in database")
            return jsonify({
                'success': False,
                'error': 'Photo not found in database'
            }), 404
        
        print(f"✅ Photo found: {photo_data.get('filename')}")
        
        # Check if file is in Dropbox
        dropbox_path = photo_data.get('dropbox_path')
        if not dropbox_path:
            print(f"❌ No Dropbox path for this photo")
            return jsonify({
                'success': False,
                'error': 'File not stored in Dropbox'
            }), 404
        
        print(f"📍 Dropbox path: {dropbox_path}")
        
        # Check if Dropbox is connected
        if not dropbox_manager.dbx:
            print(f"❌ Dropbox not connected")
            return jsonify({
                'success': False,
                'error': 'Dropbox service not available'
            }), 503
        
        print(f"✅ Dropbox connected")
        
        # Try to get temporary download link
        try:
            print(f"🔗 Getting temporary download link...")
            temp_link = dropbox_manager.get_temporary_link(dropbox_path)
            
            if temp_link:
                print(f"✅ Temporary link generated successfully")
                print(f"🌐 Redirecting to: {temp_link[:50]}...")
                print(f"{'='*60}\n")
                return redirect(temp_link)
            else:
                print(f"❌ Could not generate temporary link")
                raise Exception("Failed to generate Dropbox temporary link")
                
        except Exception as dropbox_error:
            print(f"❌ Dropbox error: {str(dropbox_error)}")
            
            # Try alternative: Download file and serve directly
            print(f"🔄 Trying alternative: Direct download from Dropbox...")
            try:
                file_stream = dropbox_manager.download_file(dropbox_path)
                
                if file_stream:
                    print(f"✅ File downloaded from Dropbox successfully")
                    print(f"{'='*60}\n")
                    
                    filename = photo_data.get('filename', f'photo-{photo_id}.jpg')
                    return send_file(
                        file_stream,
                        mimetype='image/jpeg',
                        as_attachment=True,
                        download_name=filename
                    )
                else:
                    raise Exception("Failed to download file from Dropbox")
                    
            except Exception as download_error:
                print(f"❌ Direct download failed: {str(download_error)}")
                raise
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ DOWNLOAD ERROR: {error_msg}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'photo_id': photo_id,
            'details': 'Check server logs for more information'
        }), 500
    
@app.route('/api/photographer/delete/<photo_id>', methods=['DELETE', 'OPTIONS'])
@cross_origin()
def delete_photo(photo_id):
    """Delete photo from database, local storage, and Dropbox"""
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
                print(f"☁️  Deleted from Dropbox: {photo_data['dropbox_path']}")
            except Exception as e:
                print(f"⚠️  Could not delete from Dropbox: {e}")
        
        # Delete compressed file
        compressed_deleted = False
        if photo_data.get('path_compressed') and os.path.exists(photo_data['path_compressed']):
            try:
                os.remove(photo_data['path_compressed'])
                compressed_deleted = True
                print(f"📦 Deleted compressed: {photo_data['path_compressed']}")
            except Exception as e:
                print(f"⚠️  Could not delete compressed file: {e}")
        
        # Delete face embeddings
        faces_data = photo_data.get('faces_data', [])
        for face in faces_data:
            embedding_id = face.get('embedding_id')
            if embedding_id in face_system.photo_database['face_embeddings']:
                del face_system.photo_database['face_embeddings'][embedding_id]
        
        # Delete from database
        del face_system.photo_database['photos'][photo_id]
        face_system.save_database()
        
        print(f"✅ Deleted photo: {photo_data['filename']}")
        
        return jsonify({
            'success': True,
            'message': 'Photo deleted successfully',
            'dropbox_deleted': dropbox_deleted,
            'compressed_deleted': compressed_deleted,
            'embeddings_deleted': len(faces_data)
        })
        
    except Exception as e:
        print(f"❌ Delete error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Soccer Clinic Face Recognition (Optimized Storage)',
        'total_photos': len(face_system.photo_database['photos']),
        'total_faces': len(face_system.photo_database['face_embeddings']),
        'dropbox_connected': dropbox_manager.dbx is not None,
        'storage': {
            'compressed': UPLOAD_COMPRESSED,
            'strategy': 'Original to Dropbox + Compressed for Preview'
        }
    })

# ==================== BATCH PROCESSING ====================

def batch_process_from_folder(folder_path):
    """Batch process images - upload directly to Dropbox"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    if not dropbox_manager.dbx:
        print("❌ Dropbox not connected! Cannot proceed with batch upload.")
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
    print(f"⚽ BATCH PROCESSING - DIRECT TO DROPBOX")
    print(f"{'='*70}")
    print(f"📁 Folder: {folder_path}")
    print(f"📸 Total Images: {total}")
    print(f"☁️  Strategy: Direct upload to Dropbox (NO local original)")
    print(f"{'='*70}\n")
    
    confirm = input(f"Process {total} images? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Cancelled")
        return
    
    processed = 0
    failed = 0
    uploaded_to_dropbox = 0
    compressed_count = 0
    
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
            
            # Face detection on image array
            faces_data = face_system.detect_faces_from_image_array(img)
            
            # Convert to bytes
            _, buffer = cv2.imencode(ext, img)
            image_bytes = buffer.tobytes()
            
            # Upload to Dropbox directly
            dropbox_path = f"{DROPBOX_FOLDER}/{unique_filename}"
            dropbox_result = dropbox_manager.upload_from_memory(image_bytes, dropbox_path)
            
            if not dropbox_result['success']:
                print(f"❌ Dropbox upload failed")
                failed += 1
                continue
            
            uploaded_to_dropbox += 1
            
            # Create compressed version
            temp_path = os.path.join(UPLOAD_TEMP, unique_filename)
            cv2.imwrite(temp_path, img)
            
            compressed_path = os.path.join(UPLOAD_COMPRESSED, unique_filename)
            compress_success = compressor.compress_for_preview(
                temp_path,
                compressed_path,
                max_size_kb=100
            )
            
            # Delete temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if compress_success:
                compressed_count += 1
            
            # Save to database
            photo_id = str(uuid.uuid4())
            face_system.photo_database['photos'][photo_id] = {
                'path_original': None,  # ✅ NO LOCAL ORIGINAL
                'path_compressed': compressed_path if compress_success else None,
                'filename': unique_filename,
                'faces_data': faces_data,
                'metadata': metadata.copy(),
                'uploaded_at': datetime.now().isoformat(),
                'dropbox_path': dropbox_path,
                'dropbox_link': dropbox_result.get('shared_link', None)
            }
            
            processed += 1
            print(f"✅ {len(faces_data)} faces | ☁️ Dropbox")
            
            # Save checkpoint every 10 images
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
    print(f"✅ BATCH PROCESSING COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Processed: {processed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"☁️  Uploaded to Dropbox: {uploaded_to_dropbox}/{processed}")
    print(f"📦 Compressed: {compressed_count}/{processed}")
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
            print("⚽ SOCCER CLINIC - BATCH PROCESSING (OPTIMIZED STORAGE)")
            print("="*70)
            batch_process_from_folder(folder_path)
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("\n" + "="*70)
            print("⚽ SOCCER CLINIC - AI FACE RECOGNITION (OPTIMIZED STORAGE)")
            print("="*70)
            print("\n📖 USAGE:")
            print("  python app.py                    # Run web server")
            print("  python app.py --batch <folder>   # Batch process with optimized storage")
            print("  python app.py --help             # Show this help")
            print("\n💾 STORAGE STRATEGY:")
            print("  • Original files → Dropbox (cloud backup)")
            print("  • Compressed files → Local server (fast preview ~100KB)")
            print("  • Face recognition on original quality")
            print("  • Download serves original from Dropbox")
            print("  • Preview serves compressed from local")
            print("\n🎯 BENEFITS:")
            print("  • 10x faster page loading")
            print("  • Reduced server load & bandwidth")
            print("  • Original quality preserved")
            print("  • Optimal user experience")
            print("="*70 + "\n")
            sys.exit(0)
        else:
            print("❌ Invalid arguments. Use --help for usage information.")
            sys.exit(1)
    
    if not dropbox_manager.dbx:
        print("\n" + "="*70)
        print("⚠️  WARNING: DROPBOX NOT CONNECTED")
        print("="*70)
        print("The app will work but original files won't be backed up to Dropbox.")
        print("Set DROPBOX_ACCESS_TOKEN in the code to enable Dropbox.")
        print("="*70 + "\n")
        
        proceed = input("Continue without Dropbox? (y/n): ").lower()
        if proceed != 'y':
            print("❌ Exiting...")
            sys.exit(1)
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*70)
    print("⚽ SOCCER CLINIC - AI FACE RECOGNITION (OPTIMIZED STORAGE)")
    print("="*70)
    print(f"\n☁️  Dropbox Status: {'✅ Connected' if dropbox_manager.dbx else '❌ Disconnected'}")
    print(f"💾 Storage Strategy:")
    print(f"   📦 Compressed → {UPLOAD_COMPRESSED} (Preview)")
    
    use_https = input("\nUse HTTPS? (y/n) [recommended for mobile camera]: ").lower() == 'y'
    
    if use_https:
        has_ssl = generate_self_signed_cert()
        
        if has_ssl:
            print(f"\n🌐 HTTPS URLs:")
            print(f"   Laptop/Desktop: https://localhost:4000")
            print(f"   Mobile (WiFi):  https://{local_ip}:4000")
            print("\n" + "="*70)
            print("✨ OPTIMIZED FEATURES:")
            print("   📦 Fast preview (~100KB compressed)")
            print("   ☁️  Original quality in Dropbox")
            print("   🔍 Face recognition on full quality")
            print("   ⬇️  Download original from cloud")
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
        print("✨ OPTIMIZED FEATURES:")
        print("   📦 Fast preview (~100KB compressed)")
        print("   ☁️  Original quality in Dropbox")
        print("   🔍 Face recognition on full quality")
        print("   ⬇️  Download original from cloud")
        print("="*70 + "\n")
        
        app.run(host='0.0.0.0', port=4000, debug=False, threaded=True)     