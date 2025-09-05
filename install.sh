#!/usr/bin/env bash

set -e  # Detener en caso de error

echo "🚀 Configurando entorno Multi-GPU Llama.cpp API"
echo "================================================"

# 1. Verificar dependencias del sistema
echo "🔍 Verificando dependencias del sistema..."

# Verificar cmake
#!/bin/bash

# Verificar cmake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake no encontrado. Instalando sin sudo..."
    
    # Crear directorio local para binarios si no existe
    mkdir -p ~/bin
    mkdir -p ~/local
    
    # Método 1: Descargar binario precompilado de CMake
    echo "📥 Descargando CMake precompilado..."
    cd ~/local
    
    # Obtener la última versión estable (ajustar según necesidad)
    CMAKE_VERSION="3.27.7"
    CMAKE_ARCHIVE="cmake-${CMAKE_VERSION}-linux-x86_64"
    
    # Descargar si no existe
    if [ ! -f "${CMAKE_ARCHIVE}.tar.gz" ]; then
        wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_ARCHIVE}.tar.gz"
    fi
    
    # Extraer
    if [ ! -d "$CMAKE_ARCHIVE" ]; then
        tar -xzf "${CMAKE_ARCHIVE}.tar.gz"
    fi
    
    # Crear enlace simbólico en ~/bin
    ln -sf ~/local/${CMAKE_ARCHIVE}/bin/cmake ~/bin/cmake
    ln -sf ~/local/${CMAKE_ARCHIVE}/bin/ctest ~/bin/ctest
    ln -sf ~/local/${CMAKE_ARCHIVE}/bin/cpack ~/bin/cpack
    
    # Agregar ~/bin al PATH si no está
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/bin:$PATH"
    fi
    
    echo "✅ CMake instalado en ~/bin/"
    echo "🔄 Ejecuta 'source ~/.bashrc' o reinicia la terminal"
    
else
    echo "✅ CMake encontrado: $(cmake --version | head -n1)"
fi

# Verificar instalación
if command -v cmake &> /dev/null; then
    echo "✅ CMake está disponible: $(which cmake)"
    echo "📋 Versión: $(cmake --version | head -n1)"
else
    echo "⚠️  CMake instalado pero no en PATH. Ejecuta: source ~/.bashrc"
fi

# Verificar compilador C++
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "❌ Compilador C++ no encontrado"
    exit 1
else
    echo "✅ Compilador C++ disponible"
fi

# Verificar CUDA (opcional)
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo "✅ CUDA encontrado: v$CUDA_VERSION"
    USE_CUDA=true
else
    echo "⚠️  CUDA no encontrado. Compilando solo para CPU."
    USE_CUDA=false
fi

# Verificar nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi disponible"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "   📌 GPU detectada: $line"
    done
else
    echo "⚠️  nvidia-smi no disponible"
fi

echo ""

# 2. Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

echo ""

# 3. Instalar dependencias Python
echo "📥 Instalando dependencias Python..."
pip install fastapi uvicorn aiohttp aiofiles sentence-transformers torch psutil httpx huggingface-hub

# Instalar PyTorch con CUDA si está disponible
if [ "$USE_CUDA" = true ]; then
    echo "🔥 Instalando PyTorch con soporte CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "💾 Instalando PyTorch solo CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""

# 4. Clonar y compilar llama.cpp
echo "📦 Descargando y compilando llama.cpp..."

# Limpiar instalación previa si existe
if [ -d "llama.cpp" ]; then
    echo "🧹 Limpiando instalación anterior..."
    rm -rf llama.cpp
fi

# Clonar repositorio
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Crear directorio build
mkdir build
cd build

# Configurar compilación
echo "⚙️  Configurando compilación..."
if [ "$USE_CUDA" = true ]; then
    echo "   🔥 Compilando con soporte CUDA..."
    cmake .. \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA_F16=ON
else
    echo "   💾 Compilando solo para CPU..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON
fi

# Compilar
echo "🔨 Compilando llama.cpp (esto puede tardar varios minutos)..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Verificar compilación
if [ -f "bin/llama-server" ]; then
    echo "✅ llama-server compilado correctamente"
    
    # Crear symlink para compatibilidad
    cd ..  # Salir de build/
    ln -sf build/bin/llama-server server
    echo "🔗 Creado symlink: llama.cpp/server -> build/bin/llama-server"
    
    # Probar que funciona
    echo "🧪 Probando servidor..."
    if ./server --help > /dev/null 2>&1; then
        echo "✅ Servidor funciona correctamente"
    else
        echo "⚠️  Advertencia: Error probando el servidor"
    fi
else
    echo "❌ Error: No se pudo compilar llama-server"
    exit 1
fi

cd ..  # Regresar al directorio raíz

echo ""

# 5. Crear directorio para modelos
echo "📁 Creando directorio para modelos..."
mkdir -p models
echo "✅ Directorio 'models' creado"

echo ""

# 6. Información de instalación completada
echo "🎉 ¡Instalación completada exitosamente!"
echo "========================================"
echo ""
echo "📋 Resumen de la instalación:"
echo "   ✅ CMake: $(cmake --version | head -n1)"
echo "   ✅ Python deps: FastAPI, PyTorch, sentence-transformers, etc."

if [ "$USE_CUDA" = true ]; then
    echo "   ✅ llama.cpp: Compilado con soporte CUDA"
else
    echo "   ✅ llama.cpp: Compilado para CPU"
fi

echo "   ✅ Servidor: ./llama.cpp/server (symlink a build/bin/llama-server)"
echo "   ✅ Directorio modelos: ./models/"
echo ""

# 7. Instrucciones de uso
echo "🚀 Para ejecutar la API:"
echo "========================"
echo ""
echo "# Uso básico (detecta GPUs automáticamente):"
echo "python main.py"
echo ""
echo "# Con configuración personalizada:"
echo "python main.py \\"
echo "    --server-binary ./llama.cpp/server \\"
echo "    --repo-id \"unsloth/Qwen2.5-Coder-32B-Instruct-GGUF\" \\"
echo "    --model-filename \"Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf\" \\"
echo "    --context-size 32768 \\"
echo "    --max-output-tokens 4096 \\"
echo "    --parallel-requests 3 \\"
echo "    --min-vram-gb 18 \\"
echo "    --max-gpus 1"
echo ""

# 8. Información adicional
echo "📚 Información adicional:"
echo "========================="
echo "• API principal: http://localhost:3000"
echo "• Health check: http://localhost:3001/health"
echo "• Métricas: http://localhost:3001/metrics"
echo "• Logs: Se mostrarán en la consola"
echo "• Modelos se descargan en: ./models/"
echo ""

if [ "$USE_CUDA" = true ]; then
    echo "🎮 GPUs detectadas:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl -v0 -s': GPU ' | sed 's/^[ \t]*/   /'
    fi
    echo ""
fi

echo "💡 Consejos:"
echo "============"
echo "• Primera ejecución descargará el modelo (~15-20GB)"
echo "• Asegúrate de tener suficiente espacio en disco"
echo "• Para mejores resultados usa SSD"
echo "• Monitorea el uso de VRAM con nvidia-smi"
echo ""
echo "✅ ¡Todo listo! Ejecuta 'python main.py' para comenzar."