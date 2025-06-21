#!/bin/bash
# setup.sh - Professional ACX Cover Processor Setup
# GitHub: https://github.com/eblessings/Acx-processor.git

set -e

echo "🚀 Professional ACX Cover Processor Setup"
echo "📍 GitHub: https://github.com/eblessings/Acx-processor.git"
echo ""

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

# Create directories
echo "📁 Creating directories..."
mkdir -p uploads cache logs

# Copy environment file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Download EasyOCR models
echo "🤖 Downloading AI models..."
python3 -c "
import easyocr
try:
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print('✅ EasyOCR models downloaded successfully')
except Exception as e:
    print(f'⚠️  EasyOCR model download failed: {e}')
    print('   Models will be downloaded on first use')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start the server:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "🐳 Or use Docker:"
echo "   docker-compose up --build"
echo ""
echo "📖 Visit: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "💡 For your 'Tropical Americans' cover:"
echo "   1. Upload your image"
echo "   2. Select 'Text-Aware' method"
echo "   3. Choose 3000x3000 size"
echo "   4. Download your ACX-ready cover!"