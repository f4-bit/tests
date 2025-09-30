#!/bin/bash

# Script para ejecutar servicios vLLM con sistema de colas
# Uso: ./run_vllm_services.sh [opción]
# Opciones:
#   backend  - Solo inicia el backend (main_vllm.py)
#   queue    - Solo inicia el sistema de colas (queue_vllm.py)
#   all      - Inicia ambos servicios (por defecto)

set -e  # Termina si hay error

# ============= Configuración =============
BACKEND_FILE="main_vllm.py"
QUEUE_FILE="queue_vllm.py"

BACKEND_PORT=8000
BACKEND_HOST="0.0.0.0"

QUEUE_PORT=8080
QUEUE_HOST="0.0.0.0"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Variables globales para PIDs
BACKEND_PID=""
QUEUE_PID=""

# ============= Funciones =============

print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}     🚀 vLLM Services Manager v1.0           ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo -e "\n${BLUE}▶ $1${NC}"
    echo -e "${BLUE}$(printf '─%.0s' {1..50})${NC}"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}❌ Error: Archivo $1 no encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Archivo encontrado: $1"
}

check_dependencies() {
    print_section "Verificando dependencias"
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 no encontrado${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Python3: $(python3 --version)"
    
    # Verificar dependencias Python
    echo -e "${YELLOW}🔍 Verificando paquetes Python...${NC}"
    
    local missing_deps=()
    
    python3 -c "import vllm" 2>/dev/null || missing_deps+=("vllm")
    python3 -c "import fastapi" 2>/dev/null || missing_deps+=("fastapi")
    python3 -c "import uvicorn" 2>/dev/null || missing_deps+=("uvicorn")
    python3 -c "import httpx" 2>/dev/null || missing_deps+=("httpx")
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${YELLOW}⚠️  Instalando dependencias faltantes: ${missing_deps[*]}${NC}"
        pip install "${missing_deps[@]}" --quiet
        echo -e "${GREEN}✓${NC} Dependencias instaladas"
    else
        echo -e "${GREEN}✓${NC} Todas las dependencias están instaladas"
    fi
}

check_gpu() {
    print_section "Verificando GPU"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓${NC} GPU detectada:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | while IFS=',' read -r name total free; do
            echo -e "  ${CYAN}├─${NC} Modelo: $name"
            echo -e "  ${CYAN}├─${NC} Memoria total: ${total} MB"
            echo -e "  ${CYAN}└─${NC} Memoria libre: ${free} MB"
        done
    else
        echo -e "${YELLOW}⚠️  nvidia-smi no encontrado. Ejecutando sin GPU.${NC}"
    fi
}

setup_environment() {
    print_section "Configurando entorno"
    
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export TOKENIZERS_PARALLELISM=false
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    
    echo -e "${GREEN}✓${NC} Variables de entorno configuradas"
    echo -e "  ${CYAN}├─${NC} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo -e "  ${CYAN}├─${NC} TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
    echo -e "  ${CYAN}└─${NC} VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD"
}

check_port_available() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${RED}❌ Puerto $port ya está en uso${NC}"
        echo -e "${YELLOW}💡 Puedes detener el proceso con: kill \$(lsof -t -i:$port)${NC}"
        return 1
    fi
    echo -e "${GREEN}✓${NC} Puerto $port disponible"
    return 0
}

wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=60
    local attempt=0
    
    echo -e "${YELLOW}⏳ Esperando que $service_name esté listo...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://$host:$port/" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name está listo!${NC}"
            echo -e "${CYAN}📄 Docs: http://$host:$port/docs${NC}"
            return 0
        fi
        
        # Mostrar progreso cada 5 segundos
        if [ $((attempt % 5)) -eq 0 ]; then
            echo -e "${YELLOW}   Esperando... (${attempt}s / $((max_attempts * 2))s)${NC}"
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ Timeout esperando $service_name${NC}"
    return 1
}

start_backend() {
    print_section "Iniciando Backend API (main_vllm.py)"
    
    check_file_exists "$BACKEND_FILE"
    check_port_available "$BACKEND_PORT" || exit 1
    
    echo -e "${CYAN}🔧 Configuración:${NC}"
    echo -e "  ${CYAN}├─${NC} Host: $BACKEND_HOST"
    echo -e "  ${CYAN}├─${NC} Puerto: $BACKEND_PORT"
    echo -e "  ${CYAN}└─${NC} Archivo: $BACKEND_FILE"
    
    echo -e "\n${GREEN}🚀 Lanzando backend...${NC}"
    
    # Iniciar el backend
    uvicorn ${BACKEND_FILE%.*}:app \
        --host "$BACKEND_HOST" \
        --port "$BACKEND_PORT" \
        --workers 1 \
        --log-level info \
        > "backend_${BACKEND_PORT}.log" 2>&1 &
    
    BACKEND_PID=$!
    echo -e "${GREEN}✓${NC} Backend iniciado (PID: $BACKEND_PID)"
    echo -e "${CYAN}📋 Logs: backend_${BACKEND_PORT}.log${NC}"
    
    # Esperar a que el backend esté listo
    if ! wait_for_service "$BACKEND_HOST" "$BACKEND_PORT" "Backend API"; then
        echo -e "${RED}❌ Error: Backend no respondió${NC}"
        echo -e "${YELLOW}📋 Últimas líneas del log:${NC}"
        tail -n 20 "backend_${BACKEND_PORT}.log"
        cleanup
        exit 1
    fi
}

start_queue() {
    print_section "Iniciando Queue API (queue_vllm.py)"
    
    check_file_exists "$QUEUE_FILE"
    check_port_available "$QUEUE_PORT" || exit 1
    
    echo -e "${CYAN}🔧 Configuración:${NC}"
    echo -e "  ${CYAN}├─${NC} Host: $QUEUE_HOST"
    echo -e "  ${CYAN}├─${NC} Puerto: $QUEUE_PORT"
    echo -e "  ${CYAN}├─${NC} Archivo: $QUEUE_FILE"
    echo -e "  ${CYAN}└─${NC} Backend URL: http://localhost:$BACKEND_PORT"
    
    echo -e "\n${GREEN}🚀 Lanzando sistema de colas...${NC}"
    
    # Iniciar el sistema de colas
    uvicorn ${QUEUE_FILE%.*}:app \
        --host "$QUEUE_HOST" \
        --port "$QUEUE_PORT" \
        --workers 1 \
        --log-level info \
        > "queue_${QUEUE_PORT}.log" 2>&1 &
    
    QUEUE_PID=$!
    echo -e "${GREEN}✓${NC} Queue API iniciada (PID: $QUEUE_PID)"
    echo -e "${CYAN}📋 Logs: queue_${QUEUE_PORT}.log${NC}"
    
    # Esperar a que la queue API esté lista
    if ! wait_for_service "$QUEUE_HOST" "$QUEUE_PORT" "Queue API"; then
        echo -e "${RED}❌ Error: Queue API no respondió${NC}"
        echo -e "${YELLOW}📋 Últimas líneas del log:${NC}"
        tail -n 20 "queue_${QUEUE_PORT}.log"
        cleanup
        exit 1
    fi
}

cleanup() {
    echo -e "\n${YELLOW}🛑 Deteniendo servicios...${NC}"
    
    if [ ! -z "$QUEUE_PID" ]; then
        echo -e "${YELLOW}  Deteniendo Queue API (PID: $QUEUE_PID)...${NC}"
        kill $QUEUE_PID 2>/dev/null || true
        wait $QUEUE_PID 2>/dev/null || true
        echo -e "${GREEN}  ✓${NC} Queue API detenida"
    fi
    
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${YELLOW}  Deteniendo Backend API (PID: $BACKEND_PID)...${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
        echo -e "${GREEN}  ✓${NC} Backend API detenida"
    fi
    
    echo -e "${GREEN}✅ Todos los servicios detenidos${NC}"
    exit 0
}

show_status() {
    echo -e "\n${MAGENTA}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC}           📊 Estado de los Servicios          ${MAGENTA}║${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════╝${NC}"
    
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "\n${CYAN}Backend API:${NC}"
        echo -e "  ${CYAN}├─${NC} PID: $BACKEND_PID"
        echo -e "  ${CYAN}├─${NC} URL: http://$BACKEND_HOST:$BACKEND_PORT"
        echo -e "  ${CYAN}├─${NC} Docs: http://$BACKEND_HOST:$BACKEND_PORT/docs"
        echo -e "  ${CYAN}└─${NC} Logs: backend_${BACKEND_PORT}.log"
    fi
    
    if [ ! -z "$QUEUE_PID" ]; then
        echo -e "\n${CYAN}Queue API:${NC}"
        echo -e "  ${CYAN}├─${NC} PID: $QUEUE_PID"
        echo -e "  ${CYAN}├─${NC} URL: http://$QUEUE_HOST:$QUEUE_PORT"
        echo -e "  ${CYAN}├─${NC} Docs: http://$QUEUE_HOST:$QUEUE_PORT/docs"
        echo -e "  ${CYAN}└─${NC} Logs: queue_${QUEUE_PORT}.log"
    fi
    
    echo -e "\n${YELLOW}💡 Comandos útiles:${NC}"
    echo -e "  ${CYAN}├─${NC} Ver logs backend: tail -f backend_${BACKEND_PORT}.log"
    echo -e "  ${CYAN}├─${NC} Ver logs queue: tail -f queue_${QUEUE_PORT}.log"
    echo -e "  ${CYAN}└─${NC} Detener servicios: Ctrl+C"
    echo -e "\n${GREEN}Presiona Ctrl+C para detener todos los servicios${NC}"
    echo -e "${BLUE}$(printf '═%.0s' {1..50})${NC}\n"
}

# ============= Main Script =============

trap cleanup SIGINT SIGTERM

print_header

# Parsear argumentos
MODE=${1:-all}

case $MODE in
    backend)
        check_dependencies
        check_gpu
        setup_environment
        start_backend
        show_status
        wait $BACKEND_PID
        ;;
    queue)
        check_dependencies
        setup_environment
        start_queue
        show_status
        wait $QUEUE_PID
        ;;
    all)
        check_dependencies
        check_gpu
        setup_environment
        start_backend
        echo ""
        start_queue
        show_status
        wait $BACKEND_PID $QUEUE_PID
        ;;
    *)
        echo -e "${RED}❌ Opción inválida: $MODE${NC}"
        echo -e "${YELLOW}Uso: $0 [backend|queue|all]${NC}"
        exit 1
        ;;
esac