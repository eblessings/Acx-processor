# requirements.txt
opencv-contrib-python==4.8.1.78
easyocr==1.7.0
scikit-image==0.21.0
scikit-learn==1.3.0
Pillow==10.0.1
torch==2.1.0
torchvision==0.16.0
fastapi==0.103.2
uvicorn[standard]==0.23.2
python-multipart==0.0.6
numpy==1.24.3
scipy==1.11.3
psutil==5.9.5
slowapi==0.1.9
python-dotenv==1.0.0
aiofiles==23.2.1
redis==5.0.0  # For caching in production
celery==5.3.4  # For background processing

# setup.py
from setuptools import setup, find_packages

setup(
    name="acx-cover-processor",
    version="2.0.0",
    description="Professional ACX audiobook cover processing with AI-powered content analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "opencv-contrib-python>=4.8.0",
        "easyocr>=1.7.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "psutil>=5.9.0",
        "slowapi>=0.1.9",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.0.0",
    ],
    extras_require={
        "production": ["redis>=5.0.0", "celery>=5.3.0"],
        "monitoring": ["prometheus-client>=0.17.0"],
        "gpu": ["torch[gpu]", "torchvision[gpu]"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# setup.sh - Quick setup script
#!/bin/bash
set -e

echo "🚀 Setting up Professional ACX Cover Processor..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs uploads cache static

# Create environment file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment configuration..."
    cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Processing Configuration
MAX_FILE_SIZE=50000000
SUPPORTED_FORMATS=jpg,jpeg,png,gif,webp,bmp
DEFAULT_OUTPUT_SIZE=3000x3000

# Performance Settings
WORKERS=4
MAX_CONCURRENT_PROCESSES=10
ENABLE_GPU=True
LOG_LEVEL=INFO

# Storage
UPLOAD_PATH=./uploads
CACHE_PATH=./cache
CACHE_TTL=3600

# Security
SECRET_KEY=$(openssl rand -hex 32)
ALLOWED_HOSTS=localhost,127.0.0.1
EOF
fi

# Download required models (if needed)
echo "🤖 Downloading AI models..."
python3 -c "
import easyocr
try:
    reader = easyocr.Reader(['en'], gpu=False)
    print('✅ EasyOCR models downloaded successfully')
except Exception as e:
    print(f'⚠️  EasyOCR model download failed: {e}')
"

echo "🎉 Setup complete!"
echo ""
echo "To start the server:"
echo "1. source venv/bin/activate"
echo "2. python main.py"
echo ""
echo "Or use Docker:"
echo "docker-compose up --build"

# main.py - Enhanced production main file
import os
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from dotenv import load_dotenv
import aiofiles
import asyncio
from typing import List, Optional
import time
import uuid
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Import our processing classes
from intelligent_acx_system import ProfessionalACXProcessor, logger

# Configuration
class Config:
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50000000))
    UPLOAD_PATH = Path(os.getenv("UPLOAD_PATH", "./uploads"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "./cache"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))
    WORKERS = int(os.getenv("WORKERS", 4))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create directories
Config.UPLOAD_PATH.mkdir(exist_ok=True)
Config.CACHE_PATH.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI(
    title="Professional ACX Cover Processor",
    description="AI-Powered Audiobook Cover Optimization for ACX Compliance",
    version="2.0.0",
    docs_url="/docs" if Config.DEBUG else None,
    redoc_url="/redoc" if Config.DEBUG else None
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if Config.DEBUG else ["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global processor instance
processor = ProfessionalACXProcessor()

# In-memory storage for demo (use Redis in production)
processing_cache = {}
processing_stats = {
    "total_processed": 0,
    "successful": 0,
    "failed": 0,
    "total_processing_time": 0
}

class ProcessingJob:
    def __init__(self, job_id: str, filename: str):
        self.job_id = job_id
        self.filename = filename
        self.status = "queued"  # queued, processing, completed, failed
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow()
        self.completed_at = None

# Routes

@app.get("/")
async def root():
    """Serve the frontend application"""
    if Path("index.html").exists():
        return FileResponse("index.html")
    return {"message": "ACX Cover Processor API", "version": "2.0.0", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Professional ACX Processor",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    import psutil
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('.').percent,
        },
        "processing": processing_stats,
        "cache_size": len(processing_cache)
    }

@app.get("/methods")
async def get_available_methods():
    """Get available processing methods and configurations"""
    return {
        "methods": [
            {
                "id": "auto",
                "name": "Auto (AI-Recommended)",
                "description": "Automatically selects the best cropping method based on content analysis"
            },
            {
                "id": "text_aware",
                "name": "Text-Aware Cropping",
                "description": "Preserves all important text elements (title, author, subtitle)"
            },
            {
                "id": "saliency_based",
                "name": "Saliency-Based",
                "description": "Focuses on visually important regions using advanced saliency detection"
            },
            {
                "id": "composition_optimized",
                "name": "Composition-Optimized",
                "description": "Optimized for visual composition using rule of thirds and balance"
            },
            {
                "id": "multiscale",
                "name": "Multi-scale Analysis",
                "description": "Multi-scale feature analysis for complex compositions"
            }
        ],
        "sizes": ["2400x2400", "3000x3000"],
        "formats": ["jpg", "png"],
        "max_file_size": Config.MAX_FILE_SIZE,
        "supported_formats": os.getenv("SUPPORTED_FORMATS", "jpg,jpeg,png,gif,webp,bmp").split(",")
    }

@app.post("/analyze")
@limiter.limit("20/minute")
async def analyze_cover(request: Request, file: UploadFile = File(...)):
    """Analyze uploaded cover image without processing"""
    start_time = time.time()
    
    try:
        # Validate file
        if file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Read file content
        content = await file.read()
        
        # Process with analysis only
        result = processor.process_cover(content, output_size='3000x3000', method='auto')
        
        # Update stats
        processing_time = time.time() - start_time
        processing_stats["total_processed"] += 1
        processing_stats["total_processing_time"] += processing_time
        
        if result.get('success', False):
            processing_stats["successful"] += 1
            # Remove processed image data for analysis-only endpoint
            result.pop('processed_image', None)
        else:
            processing_stats["failed"] += 1
        
        logger.info(f"Analysis completed in {processing_time:.2f}s", extra={
            'processing_time': processing_time,
            'image_size': f"{len(content)} bytes",
            'method_used': 'analysis_only'
        })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        processing_stats["failed"] += 1
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@app.post("/process")
@limiter.limit("10/minute")
async def process_cover(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    size: str = "3000x3000",
    method: str = "auto",
    async_processing: bool = False
):
    """Process cover image with specified parameters"""
    
    try:
        # Validate inputs
        if size not in ["2400x2400", "3000x3000"]:
            raise HTTPException(status_code=400, detail="Invalid size. Use '2400x2400' or '3000x3000'")
        
        if file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Read file content
        content = await file.read()
        
        if async_processing:
            # Async processing for large files
            job_id = str(uuid.uuid4())
            job = ProcessingJob(job_id, file.filename)
            processing_cache[job_id] = job
            
            # Start background processing
            background_tasks.add_task(process_image_async, job_id, content, size, method)
            
            return JSONResponse(content={
                "success": True,
                "job_id": job_id,
                "status": "queued",
                "message": "Processing started. Use /status/{job_id} to check progress."
            })
        
        else:
            # Synchronous processing
            start_time = time.time()
            result = processor.process_cover(content, output_size=size, method=method)
            
            # Update stats
            processing_time = time.time() - start_time
            processing_stats["total_processed"] += 1
            processing_stats["total_processing_time"] += processing_time
            
            if result.get('success', False):
                processing_stats["successful"] += 1
            else:
                processing_stats["failed"] += 1
            
            logger.info(f"Processing completed in {processing_time:.2f}s", extra={
                'processing_time': processing_time,
                'image_size': f"{len(content)} bytes",
                'method_used': method
            })
            
            return JSONResponse(content=result)
        
    except Exception as e:
        processing_stats["failed"] += 1
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")

async def process_image_async(job_id: str, content: bytes, size: str, method: str):
    """Background processing function"""
    job = processing_cache.get(job_id)
    if not job:
        return
    
    try:
        job.status = "processing"
        job.progress = 10
        
        start_time = time.time()
        result = processor.process_cover(content, output_size=size, method=method)
        
        processing_time = time.time() - start_time
        processing_stats["total_processed"] += 1
        processing_stats["total_processing_time"] += processing_time
        
        if result.get('success', False):
            processing_stats["successful"] += 1
            job.status = "completed"
            job.result = result
        else:
            processing_stats["failed"] += 1
            job.status = "failed"
            job.error = result.get('error', 'Unknown error')
        
        job.progress = 100
        job.completed_at = datetime.utcnow()
        
    except Exception as e:
        processing_stats["failed"] += 1
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        logger.error(f"Async processing failed for job {job_id}: {e}", exc_info=True)

@app.get("/status/{job_id}")
async def get_processing_status(job_id: str):
    """Get status of async processing job"""
    job = processing_cache.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
    }
    
    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()
        response["processing_time"] = (job.completed_at - job.created_at).total_seconds()
    
    if job.result:
        response["result"] = job.result
    
    if job.error:
        response["error"] = job.error
    
    return response

@app.post("/batch")
@limiter.limit("5/hour")
async def batch_process(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    size: str = "3000x3000",
    method: str = "auto"
):
    """Batch process multiple cover images"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    batch_id = str(uuid.uuid4())
    job_ids = []
    
    for file in files:
        if file.size > Config.MAX_FILE_SIZE:
            continue  # Skip large files
        
        content = await file.read()
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id, file.filename)
        processing_cache[job_id] = job
        job_ids.append(job_id)
        
        # Start background processing
        background_tasks.add_task(process_image_async, job_id, content, size, method)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "job_ids": job_ids,
        "message": f"Batch processing started for {len(job_ids)} files"
    }

@app.get("/stats")
async def get_statistics():
    """Get processing statistics"""
    avg_time = (processing_stats["total_processing_time"] / 
                max(processing_stats["total_processed"], 1))
    
    return {
        "total_processed": processing_stats["total_processed"],
        "successful": processing_stats["successful"],
        "failed": processing_stats["failed"],
        "success_rate": (processing_stats["successful"] / 
                        max(processing_stats["total_processed"], 1)) * 100,
        "average_processing_time": round(avg_time, 2),
        "cache_size": len(processing_cache),
        "uptime": "System uptime info would go here"
    }

# Cleanup old cache entries
async def cleanup_cache():
    """Cleanup old cache entries"""
    cutoff = datetime.utcnow() - timedelta(hours=1)
    expired_jobs = [
        job_id for job_id, job in processing_cache.items()
        if job.created_at < cutoff
    ]
    
    for job_id in expired_jobs:
        del processing_cache[job_id]
    
    logger.info(f"Cleaned up {len(expired_jobs)} expired cache entries")

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Professional ACX Cover Processor starting up...")
    logger.info(f"Configuration: {Config.API_HOST}:{Config.API_PORT}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Max file size: {Config.MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Start cleanup task
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodic cache cleanup"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        await cleanup_cache()

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 Professional ACX Cover Processor shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        workers=1,  # Use 1 for development, multiple for production
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower()
    )

# docker-compose.prod.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  acx-processor:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DEBUG=False
      - WORKERS=4
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./uploads:/app/uploads
      - ./cache:/app/cache
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./frontend:/usr/share/nginx/html
      - ./ssl:/etc/nginx/ssl  # SSL certificates
    depends_on:
      - acx-processor
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  grafana_data: