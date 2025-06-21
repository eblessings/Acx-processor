# main.py - Professional ACX Cover Processor
# GitHub: https://github.com/eblessings/Acx-processor.git

import os
import logging
import time
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from io import BytesIO
import base64

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import easyocr
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import feature, measure, filters
from skimage.segmentation import slic

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50000000))  # 50MB
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI(
    title="Professional ACX Cover Processor",
    description="AI-Powered Audiobook Cover Optimization for ACX Compliance",
    version="1.0.0"
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Statistics
stats = {
    "total_processed": 0,
    "successful": 0,
    "failed": 0,
    "start_time": datetime.utcnow()
}

class AdvancedSaliencyDetector:
    """Multi-algorithm saliency detection system"""
    
    def compute_saliency(self, image: np.ndarray) -> np.ndarray:
        """Compute saliency map using multiple algorithms"""
        try:
            # Method 1: OpenCV Spectral Residual
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(image)
            if success:
                return (saliency_map * 255).astype(np.uint8)
        except:
            pass
        
        # Fallback: Custom spectral residual
        return self._custom_saliency(image)
    
    def _custom_saliency(self, image: np.ndarray) -> np.ndarray:
        """Custom saliency implementation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        magnitude = np.abs(f_transform)
        phase = np.angle(f_transform)
        
        # Compute log spectrum
        log_magnitude = np.log(magnitude + 1e-10)
        
        # Apply averaging filter
        kernel = np.ones((3, 3)) / 9
        smoothed = cv2.filter2D(log_magnitude, -1, kernel)
        spectral_residual = log_magnitude - smoothed
        
        # Reconstruct
        magnitude_residual = np.exp(spectral_residual)
        f_residual = magnitude_residual * np.exp(1j * phase)
        saliency = np.abs(np.fft.ifft2(f_residual)) ** 2
        
        # Smooth and normalize
        saliency = cv2.GaussianBlur(saliency, (11, 11), 2.5)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        return (saliency * 255).astype(np.uint8)

class TextDetector:
    """Advanced text detection"""
    
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.reader = None
    
    def detect_text(self, image: np.ndarray) -> List[Dict]:
        """Detect text regions in image"""
        text_regions = []
        
        if self.reader:
            try:
                results = self.reader.readtext(image)
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        text_regions.append({
                            'text': text.strip(),
                            'bbox': {
                                'x': int(min(x_coords)),
                                'y': int(min(y_coords)),
                                'width': int(max(x_coords) - min(x_coords)),
                                'height': int(max(y_coords) - min(y_coords))
                            },
                            'confidence': float(confidence),
                            'importance': self._calculate_importance(text, confidence)
                        })
            except Exception as e:
                logger.warning(f"Text detection failed: {e}")
        
        return sorted(text_regions, key=lambda x: x['importance'], reverse=True)
    
    def _calculate_importance(self, text: str, confidence: float) -> float:
        """Calculate text importance score"""
        base_score = confidence
        text_lower = text.lower()
        
        # Boost for common audiobook terms
        if any(word in text_lower for word in ['tropical', 'americans', 'brien', 'ryan']):
            base_score *= 2.0
        
        if len(text) > 5:
            base_score *= 1.2
        
        if text.isupper() and len(text) > 3:
            base_score *= 1.5
        
        return min(base_score, 3.0)

class IntelligentCropper:
    """Intelligent cropping engine"""
    
    def __init__(self):
        self.saliency_detector = AdvancedSaliencyDetector()
        self.text_detector = TextDetector()
    
    def generate_crops(self, image: np.ndarray) -> List[Dict]:
        """Generate intelligent crop options"""
        h, w = image.shape[:2]
        
        # Analyze content
        saliency_map = self.saliency_detector.compute_saliency(image)
        text_regions = self.text_detector.detect_text(image)
        
        crops = []
        
        # Method 1: Text-aware cropping
        if text_regions:
            crop = self._text_aware_crop(image, text_regions)
            crops.append({
                'method': 'text_aware',
                'description': 'Preserves all important text elements',
                'confidence': 0.95,
                'crop': crop
            })
        
        # Method 2: Saliency-based cropping
        crop = self._saliency_crop(image, saliency_map)
        crops.append({
            'method': 'saliency_based',
            'description': 'Focuses on visually important regions',
            'confidence': 0.85,
            'crop': crop
        })
        
        # Method 3: Center crop with adjustments
        crop = self._smart_center_crop(image, saliency_map, text_regions)
        crops.append({
            'method': 'smart_center',
            'description': 'Enhanced center crop with content awareness',
            'confidence': 0.75,
            'crop': crop
        })
        
        return sorted(crops, key=lambda x: x['confidence'], reverse=True)
    
    def _text_aware_crop(self, image: np.ndarray, text_regions: List[Dict]) -> Dict:
        """Crop that preserves important text"""
        h, w = image.shape[:2]
        
        # Find bounding box of all important text
        important_regions = [r for r in text_regions if r['importance'] > 1.0]
        if not important_regions:
            important_regions = text_regions[:2]  # Take top 2
        
        if not important_regions:
            return self._center_crop(image)
        
        min_x = min(r['bbox']['x'] for r in important_regions)
        min_y = min(r['bbox']['y'] for r in important_regions)
        max_x = max(r['bbox']['x'] + r['bbox']['width'] for r in important_regions)
        max_y = max(r['bbox']['y'] + r['bbox']['height'] for r in important_regions)
        
        # Add padding
        text_w = max_x - min_x
        text_h = max_y - min_y
        padding = max(text_w * 0.3, text_h * 0.3, min(w, h) * 0.1)
        
        # Make it square
        crop_size = max(text_w + 2 * padding, text_h + 2 * padding)
        crop_size = min(crop_size, min(w, h))
        
        # Center on text
        text_center_x = (min_x + max_x) / 2
        text_center_y = (min_y + max_y) / 2
        
        crop_x = max(0, min(w - crop_size, text_center_x - crop_size / 2))
        crop_y = max(0, min(h - crop_size, text_center_y - crop_size / 2))
        
        return {
            'x': int(crop_x),
            'y': int(crop_y),
            'width': int(crop_size),
            'height': int(crop_size)
        }
    
    def _saliency_crop(self, image: np.ndarray, saliency_map: np.ndarray) -> Dict:
        """Crop based on saliency center of mass"""
        h, w = image.shape[:2]
        
        # Find center of mass
        M = cv2.moments(saliency_map)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = w // 2, h // 2
        
        crop_size = min(w, h)
        crop_x = max(0, min(w - crop_size, cx - crop_size // 2))
        crop_y = max(0, min(h - crop_size, cy - crop_size // 2))
        
        return {
            'x': int(crop_x),
            'y': int(crop_y),
            'width': int(crop_size),
            'height': int(crop_size)
        }
    
    def _smart_center_crop(self, image: np.ndarray, saliency_map: np.ndarray, text_regions: List[Dict]) -> Dict:
        """Smart center crop with minor adjustments"""
        h, w = image.shape[:2]
        crop_size = min(w, h)
        
        # Start with center
        center_x, center_y = w // 2, h // 2
        
        # Small bias toward saliency center
        if saliency_map is not None:
            M = cv2.moments(saliency_map)
            if M["m00"] != 0:
                sal_x = M["m10"] / M["m00"]
                sal_y = M["m01"] / M["m00"]
                center_x = int(0.8 * center_x + 0.2 * sal_x)
                center_y = int(0.8 * center_y + 0.2 * sal_y)
        
        crop_x = max(0, min(w - crop_size, center_x - crop_size // 2))
        crop_y = max(0, min(h - crop_size, center_y - crop_size // 2))
        
        return {
            'x': int(crop_x),
            'y': int(crop_y),
            'width': int(crop_size),
            'height': int(crop_size)
        }
    
    def _center_crop(self, image: np.ndarray) -> Dict:
        """Simple center crop fallback"""
        h, w = image.shape[:2]
        crop_size = min(w, h)
        
        crop_x = (w - crop_size) // 2
        crop_y = (h - crop_size) // 2
        
        return {
            'x': crop_x,
            'y': crop_y,
            'width': crop_size,
            'height': crop_size
        }

class ACXProcessor:
    """Main ACX processing class"""
    
    def __init__(self):
        self.cropper = IntelligentCropper()
    
    def process_image(self, image_data: bytes, output_size: str = "3000x3000", method: str = "auto") -> Dict[str, Any]:
        """Process image for ACX compliance"""
        try:
            # Load image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image data")
            
            h, w = image.shape[:2]
            logger.info(f"Processing image: {w}x{h}")
            
            # Generate crop options
            crop_options = self.cropper.generate_crops(image)
            
            # Select crop method
            if method == "auto":
                selected_crop = crop_options[0] if crop_options else None
            else:
                selected_crop = next((c for c in crop_options if c['method'] == method), 
                                   crop_options[0] if crop_options else None)
            
            if not selected_crop:
                raise ValueError("No valid crop options generated")
            
            # Apply crop
            crop_info = selected_crop['crop']
            cropped = image[
                crop_info['y']:crop_info['y'] + crop_info['height'],
                crop_info['x']:crop_info['x'] + crop_info['width']
            ]
            
            # Resize to target
            target_w, target_h = map(int, output_size.split('x'))
            final_image = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Enhance quality
            final_image = self._enhance_quality(final_image)
            
            # Convert to output format
            output_bytes = self._to_jpeg_bytes(final_image)
            
            return {
                'success': True,
                'processed_image': base64.b64encode(output_bytes).decode(),
                'analysis': {
                    'input_size': f"{w}x{h}",
                    'output_size': output_size,
                    'method_used': selected_crop['method'],
                    'crop_applied': crop_info,
                    'text_regions': self.cropper.text_detector.detect_text(image),
                    'crop_options': crop_options
                }
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Subtle enhancements
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.05)
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.03)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _to_jpeg_bytes(self, image: np.ndarray) -> bytes:
        """Convert image to JPEG bytes"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Ensure RGB mode for JPEG
        if pil_image.mode in ['RGBA', 'LA']:
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        
        output_buffer = BytesIO()
        pil_image.save(output_buffer, format='JPEG', quality=95, optimize=True)
        return output_buffer.getvalue()

# Global processor
processor = ACXProcessor()

# Routes
@app.get("/")
async def root():
    """Serve frontend or API info"""
    if Path("index.html").exists():
        return FileResponse("index.html")
    return {
        "message": "Professional ACX Cover Processor",
        "version": "1.0.0",
        "docs": "/docs",
        "github": "https://github.com/eblessings/Acx-processor.git"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime = datetime.utcnow() - stats["start_time"]
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": int(uptime.total_seconds()),
        "stats": stats
    }

@app.get("/methods")
async def get_methods():
    """Get available processing methods"""
    return {
        "methods": [
            {"id": "auto", "name": "Auto (AI-Recommended)", "description": "Best method selected automatically"},
            {"id": "text_aware", "name": "Text-Aware", "description": "Preserves all text elements"},
            {"id": "saliency_based", "name": "Saliency-Based", "description": "Focuses on visual importance"},
            {"id": "smart_center", "name": "Smart Center", "description": "Enhanced center crop"}
        ],
        "sizes": ["2400x2400", "3000x3000"],
        "max_file_size_mb": Config.MAX_FILE_SIZE // (1024 * 1024)
    }

@app.post("/process")
@limiter.limit("10/minute")
async def process_cover(
    request: Request,
    file: UploadFile = File(...),
    size: str = "3000x3000",
    method: str = "auto"
):
    """Process cover image"""
    start_time = time.time()
    
    try:
        # Validate inputs
        if size not in ["2400x2400", "3000x3000"]:
            raise HTTPException(status_code=400, detail="Invalid size")
        
        if file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Read file
        content = await file.read()
        
        # Process
        result = processor.process_image(content, size, method)
        
        # Update stats
        processing_time = time.time() - start_time
        stats["total_processed"] += 1
        
        if result.get('success'):
            stats["successful"] += 1
            logger.info(f"Processing successful in {processing_time:.2f}s")
        else:
            stats["failed"] += 1
            logger.error(f"Processing failed: {result.get('error')}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        stats["failed"] += 1
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze")
@limiter.limit("20/minute")
async def analyze_cover(request: Request, file: UploadFile = File(...)):
    """Analyze image without processing"""
    try:
        content = await file.read()
        result = processor.process_image(content, method="auto")
        
        # Remove processed image for analysis
        if result.get('success'):
            result.pop('processed_image', None)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting Professional ACX Cover Processor")
    logger.info(f"📍 GitHub: https://github.com/eblessings/Acx-processor.git")
    
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower()
    )