# RunPod Optimized Dockerfile for FireRedTTS2
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .

# Install PyTorch with CUDA support
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
RUN pip install -r requirements.txt

# Install the package in development mode
RUN pip install -e .

# Install additional dependencies for enhanced functionality
RUN pip install \
    websockets \
    aiohttp \
    asyncio \
    numpy \
    scipy \
    librosa \
    soundfile \
    webrtcvad \
    openai-whisper \
    fastapi \
    uvicorn \
    python-multipart \
    redis \
    psutil

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/models \
    /workspace/cache \
    /workspace/logs \
    /workspace/uploads \
    /workspace/sessions

# Set permissions
RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true

# Create model download script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Downloading FireRedTTS2 models..."\n\
if [ ! -d "/workspace/models/FireRedTTS2" ]; then\n\
    echo "Models not found in persistent volume, downloading..."\n\
    cd /workspace/models\n\
    git lfs install\n\
    git clone https://huggingface.co/FireRedTeam/FireRedTTS2\n\
    echo "Models downloaded successfully"\n\
else\n\
    echo "Models found in persistent volume, skipping download"\n\
fi\n\
' > /workspace/download_models.sh && chmod +x /workspace/download_models.sh

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting FireRedTTS2 RunPod deployment..."\n\
\n\
# Download models if not present\n\
/workspace/download_models.sh\n\
\n\
# Set CUDA device\n\
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}\n\
\n\
# Start the enhanced web interface\n\
echo "Starting enhanced web interface on port 7860..."\n\
python3 enhanced_gradio_demo.py --pretrained-dir "/workspace/models/FireRedTTS2" --host 0.0.0.0 --port 7860\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Expose port for Gradio interface
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Set default command
CMD ["/workspace/start.sh"]