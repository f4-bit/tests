#!/bin/bash

# install.sh - Instalación simple para Unsloth Batch Inference API

echo "🚀 Instalando dependencias..."
export TOKENIZERS_PARALLELISM=false

# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch
echo "📦 Instalando PyTorch..."
#pip uninstall -y torch torchvision torchaudio
#pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# Ya tiene una versión de pytorch instalada

# Instalar FastAPI y servidor
echo "🌐 Instalando FastAPI..."
pip install fastapi uvicorn[standard]

# Instalar dependencias ML
echo "🤖 Instalando dependencias de ML..."
pip install transformers accelerate bitsandbytes

# Instalar flash attention
echo "⚡ Instalando Flash Attention..."
pip install "flash-attn>=2.5.6,<2.6.0" --use-pep517 --no-build-isolation

# Instalar utilidades
echo "🔧 Instalando utilidades..."
pip install pydantic python-multipart

echo "✅ ¡Instalación completada!"