#!/bin/bash

# install.sh - Instalación simple para Unsloth Batch Inference API

export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="+dynamo"

echo "🚀 Instalando dependencias..."

# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch
echo "📦 Instalando PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade torch

# Instalar FastAPI y servidor
echo "🌐 Instalando FastAPI..."
pip install fastapi uvicorn[standard]

# Instalar dependencias ML
echo "🤖 Instalando dependencias de ML..."
pip install transformers accelerate bitsandbytes

# Instalar Unsloth
echo "⚡ Instalando Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Instalar utilidades
echo "🔧 Instalando utilidades..."
pip install pydantic python-multipart

echo "✅ ¡Instalación completada!"
python main.py