#!/usr/bin/env bash

# 3. Actualizar pip
echo "⬆️  Actualizando pip"
pip install --upgrade pip

# 4. Instalar dependencias
echo "📥 Instalando FastAPI, Uvicorn y llama-cpp-python"
pip install fastapi uvicorn aiohttp aiofiles sentence-transformers torch psutil httpx

echo "Instalando hf hub"
pip install huggingface-hub

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make server

# --- Opción CPU ---
#pip install llama-cpp-python

# --- Opción GPU NVIDIA (comenta la línea anterior y descomenta esta si quieres usar GPU CUDA 12.4) ---
#pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# 5. Listo
echo "✅ Instalación completada!"