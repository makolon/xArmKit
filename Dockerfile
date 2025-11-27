FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    ninja-build \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Intel RealSense SDK (Ubuntu 22.04 = jammy)
RUN apt-get update && \
    mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo jammy main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && \
    apt-get install -y \
        librealsense2-utils \
        librealsense2-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install PyTorch with CUDA 11.8 support FIRST (before other packages)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/
# Install NumPy 1.x first to avoid compatibility issues
RUN pip3 install --no-cache-dir "numpy>=1.24.0,<2.0.0"
RUN pip3 install --no-cache-dir -r requirements.txt

# Install transformers with compatible huggingface-hub version
RUN pip3 install --no-cache-dir \
    "transformers>=4.30.0,<4.40.0" \
    "huggingface-hub>=0.14.0,<1.0" \
    tokenizers \
    sentencepiece

# Install additional dependencies for GroundingDINO
RUN pip3 install --no-cache-dir \
    opencv-python \
    pyrealsense2 \
    supervision \
    timm \
    addict \
    yapf \
    pycocotools

# Install dependencies for Any6D and FoundationPose
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /workspace/

# Fix git permissions for GroundingDINO
RUN git config --global --add safe.directory /workspace/third_party/groundingdino || true

# Build GroundingDINO CUDA extensions
WORKDIR /workspace/third_party/groundingdino
RUN TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6" pip3 install --no-cache-dir -e . || \
    echo "GroundingDINO CUDA extension build failed, will use CPU fallback"

# Build FoundationPose C++ extensions for Any6D
WORKDIR /workspace/third_party/any6d/foundationpose
RUN bash build_all.sh || echo "FoundationPose build failed, some features may not work"

# Set environment variables
ENV PYTHONPATH=/workspace/src:/workspace:$PYTHONPATH
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV CUDA_VISIBLE_DEVICES=0

# Create cache directory
RUN mkdir -p /workspace/.cache/huggingface

WORKDIR /workspace

# Set default command
CMD ["/bin/bash"]
