#!/bin/bash

# Script para ejecutar servidor vLLM con FastAPI
# Uso: ./run_vllm_server.sh [puerto] [host]

echo -e "${GREEN}🚀 Instalando dependencias...${NC}"
pip install vllm fastapi uvicorn
echo -e "${GREEN}✅ Dependencias instaladas${NC}"

set -e  # Termina si hay error

# Configuración por defecto
DEFAULT_PORT=8001
DEFAULT_HOST="0.0.0.0"
PYTHON_FILE="queue_vllm.py"  # Cambia por el nombre de tu archivo .py

# Parsear argumentos
PORT=${1:-$DEFAULT_PORT}
HOST=${2:-$DEFAULT_HOST}

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Iniciando servidor vLLM...${NC}"
echo -e "${YELLOW}Host: $HOST${NC}"
echo -e "${YELLOW}Puerto: $PORT${NC}"
echo -e "${YELLOW}Archivo: $PYTHON_FILE${NC}"

# Verificar que el archivo Python existe
if [ ! -f "$PYTHON_FILE" ]; then
    echo -e "${RED}❌ Error: Archivo $PYTHON_FILE no encontrado${NC}"
    exit 1
fi

# Verificar dependencias críticas
echo -e "${YELLOW}🔍 Verificando dependencias...${NC}"
python3 -c "import vllm, fastapi, uvicorn" 2>/dev/null || {
    echo -e "${RED}❌ Error: Faltan dependencias. Instala: pip install vllm fastapi uvicorn${NC}"
    exit 1
}

# Verificar GPU disponible (opcional)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ GPU detectada:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
else
    echo -e "${YELLOW}⚠️ nvidia-smi no encontrado. ¿Tienes GPU NVIDIA?${NC}"
fi

# Configurar variables de entorno para optimización
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # GPU por defecto
export TOKENIZERS_PARALLELISM=false  # Evita warnings de tokenizers
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # Para estabilidad

# Función para manejar señales de interrupción
cleanup() {
    echo -e "\n${YELLOW}🛑 Deteniendo servidor...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${GREEN}🔥 Lanzando servidor en http://$HOST:$PORT${NC}"
echo -e "${YELLOW}Presiona Ctrl+C para detener${NC}"
echo "=================================="

# Ejecutar con uvicorn directamente (más control que python -m)
uvicorn ${PYTHON_FILE%.*}:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 5000 \
    --access-log \
    --log-level info &

SERVER_PID=$!

# Esperar a que el servidor esté listo
echo -e "${YELLOW}⏳ Esperando que el servidor esté listo...${NC}"
for i in {1..30}; do
    if curl -s "http://$HOST:$PORT/docs" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Servidor listo! Documentación en: http://$HOST:$PORT/docs${NC}"
        break
    fi
    sleep 2
#    if [ $i -eq 30 ]; then
#        echo -e "${RED}❌ Timeout: El servidor tardó demasiado en iniciar${NC}"
#        kill $SERVER_PID 2>/dev/null || true
#        exit 1
#    fi
done

# Mantener el script corriendo
wait $SERVER_PID