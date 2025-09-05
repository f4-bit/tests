import asyncio
import aiohttp
import aiofiles
import argparse
import json
import logging
import os
import subprocess
import time
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from sentence_transformers import SentenceTransformer
import torch
import httpx
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Modelos Pydantic para la API
# ============================================================================

class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="El prompt para generar")
    max_tokens: int = Field(default=4096, ge=1, le=8192, description="Máximo tokens a generar")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperatura de muestreo")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=40, ge=1, le=100, description="Top-k sampling")
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0, description="Penalidad por repetición")
    stop: Optional[List[str]] = Field(default=None, description="Tokens de parada")
    stream: bool = Field(default=False, description="Streaming response")

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Mensajes del chat")
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    stop: Optional[List[str]] = Field(default=None)
    stream: bool = Field(default=False)

class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="Texto para generar embeddings")
    model: str = Field(default="all-MiniLM-L6-v2", description="Modelo de embeddings")

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    usage: Dict[str, int]

# ============================================================================
# Gestión de GPUs y Distribución
# ============================================================================

@dataclass
class GPUInfo:
    id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    utilization: int   # %
    temperature: int   # C

class GPUManager:
    """Gestiona información y distribución de GPUs"""
    
    def __init__(self):
        self.gpus: List[GPUInfo] = []
        self.refresh_gpu_info()
    
    def refresh_gpu_info(self):
        """Actualiza información de GPUs disponibles"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        values = [v.strip() for v in line.split(',')]
                        self.gpus.append(GPUInfo(
                            id=int(values[0]),
                            name=values[1],
                            memory_total=int(values[2]),
                            memory_free=int(values[3]),
                            utilization=int(values[4]),
                            temperature=int(values[5])
                        ))
                        
                logger.info(f"Detectadas {len(self.gpus)} GPUs")
                for gpu in self.gpus:
                    logger.info(f"  GPU {gpu.id}: {gpu.name} - {gpu.memory_free}/{gpu.memory_total} MB libre")
            else:
                logger.warning("No se pudieron detectar GPUs NVIDIA")
                
        except Exception as e:
            logger.error(f"Error detectando GPUs: {e}")
    
    def get_available_gpus(self, min_memory_gb: float = 18.0) -> List[int]:
        """Obtiene GPUs con suficiente memoria libre"""
        min_memory_mb = min_memory_gb * 1024
        available = [
            gpu.id for gpu in self.gpus 
            if gpu.memory_free >= min_memory_mb
        ]
        return available
    
    def get_best_gpu(self, exclude: List[int] = None) -> Optional[int]:
        """Obtiene la GPU con más memoria libre"""
        exclude = exclude or []
        available_gpus = [gpu for gpu in self.gpus if gpu.id not in exclude]
        
        if not available_gpus:
            return None
            
        best_gpu = max(available_gpus, key=lambda g: g.memory_free)
        return best_gpu.id

# ============================================================================
# Descarga de Modelos - FIXED VERSION
# ============================================================================

class ModelDownloader:
    """Descarga modelos de Hugging Face usando huggingface_hub oficial"""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    async def download_model(self, repo_id: str, filename: str) -> Path:
        """Descarga un modelo específico de HuggingFace usando huggingface_hub"""
        try:
            logger.info(f"Verificando modelo: {repo_id}/{filename}")
            
            # Usar el método oficial de huggingface_hub que maneja redirecciones automáticamente
            model_path = await asyncio.to_thread(
                self._sync_download_model, 
                repo_id, 
                filename
            )
            
            logger.info(f"Modelo disponible en: {model_path}")
            return Path(model_path)
            
        except HfHubHTTPError as e:
            logger.error(f"Error de Hugging Face Hub: {e}")
            if e.response.status_code == 404:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Modelo no encontrado: {repo_id}/{filename}"
                )
            elif e.response.status_code == 401:
                raise HTTPException(
                    status_code=401, 
                    detail="Token de Hugging Face requerido para este modelo privado"
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error descargando modelo: HTTP {e.response.status_code}"
                )
        except Exception as e:
            logger.error(f"Error descargando modelo: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error descargando modelo: {str(e)}"
            )
    
    def _sync_download_model(self, repo_id: str, filename: str) -> str:
        """Descarga síncrona usando huggingface_hub oficial"""
        try:
            logger.info(f"Descargando {filename} desde {repo_id}...")
            
            # Método principal: usar hf_hub_download
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.base_dir),
                resume_download=True,  # Reanuda descargas interrumpidas
                local_files_only=False,  # Permite descarga desde internet
            )
            
            logger.info(f"Descarga completada: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error en hf_hub_download: {e}")
            
            # Método fallback: usar snapshot_download
            try:
                logger.info("Intentando con snapshot_download como fallback...")
                
                snapshot_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(self.base_dir),
                    resume_download=True,
                    allow_patterns=[filename],  # Solo descargar el archivo específico
                )
                
                model_path = os.path.join(snapshot_path, filename)
                if os.path.exists(model_path):
                    logger.info(f"Descarga fallback completada: {model_path}")
                    return model_path
                else:
                    raise FileNotFoundError(f"Archivo {filename} no encontrado en snapshot")
                    
            except Exception as e2:
                logger.error(f"Error en snapshot_download: {e2}")
                raise e  # Re-raise el error original

    async def check_model_exists(self, model_path: str) -> bool:
        """Verifica si el modelo ya existe localmente"""
        try:
            path = Path(model_path)
            if path.exists() and path.stat().st_size > 0:
                logger.info(f"Modelo encontrado localmente: {model_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error verificando modelo: {e}")
            return False

# ============================================================================
# Servidor Llama.cpp
# ============================================================================

@dataclass
class LlamaServerConfig:
    gpu_id: int
    port: int
    model_path: Path
    context_size: int
    parallel_requests: int
    batch_size: int
    max_tokens: int

class LlamaServerManager:
    """Gestiona instancias de servidores llama.cpp en múltiples GPUs"""
    
    def __init__(self, server_binary: str = "./server"):
        self.server_binary = server_binary
        self.servers: Dict[int, subprocess.Popen] = {}  # gpu_id -> process
        self.configs: Dict[int, LlamaServerConfig] = {}
        self.base_port = 8080
        
    async def start_server(self, config: LlamaServerConfig) -> bool:
        """Inicia un servidor llama.cpp en una GPU específica"""
        gpu_id = config.gpu_id
        
        if gpu_id in self.servers:
            logger.warning(f"Servidor ya corriendo en GPU {gpu_id}")
            return True
            
        # Configurar CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        cmd = [
            self.server_binary,
            "-m", str(config.model_path),
            "--host", "0.0.0.0",
            "--port", str(config.port),
            "-c", str(config.context_size),
            "-np", str(config.parallel_requests),
            "-b", str(config.batch_size),
            "-ngl", "-1",  # Todas las capas en GPU
            "--threads", "8",
            "--log-format", "json",
            "-v"
        ]
        
        logger.info(f"Iniciando servidor GPU {gpu_id}: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.servers[gpu_id] = process
            self.configs[gpu_id] = config
            
            # Esperar a que el servidor esté listo
            await self._wait_for_server(config.port)
            logger.info(f"Servidor GPU {gpu_id} listo en puerto {config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando servidor GPU {gpu_id}: {e}")
            return False
    
    async def _wait_for_server(self, port: int, timeout: int = 60):
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    # Use localhost for internal health checks, but servers bind to 0.0.0.0
                    response = await client.get(f"http://localhost:{port}/health", timeout=2)
                    if response.status_code == 200:
                        return
            except:
                pass
            await asyncio.sleep(1)
            
        raise TimeoutError(f"Servidor en puerto {port} no respondió en {timeout}s")
    
    def stop_all_servers(self):
        """Detiene todos los servidores"""
        for gpu_id, process in self.servers.items():
            logger.info(f"Deteniendo servidor GPU {gpu_id}")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        self.servers.clear()
        self.configs.clear()
    
    def get_available_servers(self) -> List[int]:
        """Obtiene lista de servidores disponibles"""
        return list(self.servers.keys())
    
    def get_server_port(self, gpu_id: int) -> Optional[int]:
        """Obtiene el puerto de un servidor específico"""
        config = self.configs.get(gpu_id)
        return config.port if config else None

# ============================================================================
# Load Balancer
# ============================================================================

class LoadBalancer:
    """Distribuye requests entre servidores disponibles"""
    
    def __init__(self, server_manager: LlamaServerManager):
        self.server_manager = server_manager
        self.current_server = 0
        self.server_stats = {}  # gpu_id -> {"requests": int, "avg_time": float}
    
    def get_next_server(self) -> Optional[int]:
        """Obtiene el próximo servidor usando round-robin"""
        available_servers = self.server_manager.get_available_servers()
        
        if not available_servers:
            return None
        
        # Round-robin simple
        if self.current_server >= len(available_servers):
            self.current_server = 0
            
        gpu_id = available_servers[self.current_server]
        self.current_server += 1
        
        return gpu_id
    
    def get_best_server(self) -> Optional[int]:
        """Obtiene el servidor con mejor rendimiento"""
        available_servers = self.server_manager.get_available_servers()
        
        if not available_servers:
            return None
        
        # Si no hay estadísticas, usar round-robin
        if not any(server in self.server_stats for server in available_servers):
            return self.get_next_server()
        
        # Encontrar servidor con menor carga
        best_server = min(
            available_servers,
            key=lambda s: self.server_stats.get(s, {"requests": 0})["requests"]
        )
        
        return best_server
    
    def update_server_stats(self, gpu_id: int, processing_time: float):
        """Actualiza estadísticas de un servidor"""
        if gpu_id not in self.server_stats:
            self.server_stats[gpu_id] = {"requests": 0, "avg_time": 0.0}
        
        stats = self.server_stats[gpu_id]
        stats["requests"] += 1
        
        # Media móvil del tiempo de procesamiento
        alpha = 0.1
        if stats["avg_time"] == 0:
            stats["avg_time"] = processing_time
        else:
            stats["avg_time"] = alpha * processing_time + (1 - alpha) * stats["avg_time"]

# ============================================================================
# Cliente Llama.cpp
# ============================================================================

class LlamaClient:
    """Cliente para comunicarse con servidores llama.cpp"""

    def __init__(self, load_balancer: LoadBalancer, host: str = "0.0.0.0"):
        self.load_balancer = load_balancer
        self.host = host  # Allow configurable host
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutos timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_completion(self, request: CompletionRequest) -> Dict[str, Any]:
        """Genera una completion usando el mejor servidor disponible"""
        gpu_id = self.load_balancer.get_best_server()
        if gpu_id is None:
            raise HTTPException(status_code=503, detail="No hay servidores disponibles")
        
        port = self.load_balancer.server_manager.get_server_port(gpu_id)
        url = f"http://{self.host}:{port}/completion"
        
        payload = {
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repeat_penalty": request.repeat_penalty,
            "stop": request.stop or [],
            "stream": request.stream
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Error del servidor GPU {gpu_id}: {error_text}"
                    )
                
                result = await response.json()
                processing_time = time.time() - start_time
                
                # Actualizar estadísticas
                self.load_balancer.update_server_stats(gpu_id, processing_time)
                
                # Agregar metadatos
                result["gpu_id"] = gpu_id
                result["processing_time"] = processing_time
                
                return result
                
        except Exception as e:
            logger.error(f"Error comunicándose con servidor GPU {gpu_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def generate_completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Genera completion con streaming"""
        gpu_id = self.load_balancer.get_best_server()
        if gpu_id is None:
            raise HTTPException(status_code=503, detail="No hay servidores disponibles")
        
        port = self.load_balancer.server_manager.get_server_port(gpu_id)
        url = f"http://127.0.0.1:{port}/completion"
        
        payload = {
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repeat_penalty": request.repeat_penalty,
            "stop": request.stop or [],
            "stream": True
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Error del servidor GPU {gpu_id}: {error_text}"
                    )
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])
                            if 'content' in data:
                                yield f"data: {json.dumps(data)}\n\n"
                        except json.JSONDecodeError:
                            continue
                
                processing_time = time.time() - start_time
                self.load_balancer.update_server_stats(gpu_id, processing_time)
                
        except Exception as e:
            logger.error(f"Error en streaming GPU {gpu_id}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ============================================================================
# Gestor de Embeddings
# ============================================================================

class EmbeddingManager:
    """Gestiona modelos de embeddings en CPU"""
    
    def __init__(self):
        self.models: Dict[str, SentenceTransformer] = {}
        self.device = "cpu"  # Forzar CPU para embeddings
        
    async def load_model(self, model_name: str) -> SentenceTransformer:
        """Carga un modelo de embeddings"""
        if model_name not in self.models:
            logger.info(f"Cargando modelo de embeddings: {model_name}")
            
            # Forzar CPU y configurar torch
            torch.set_num_threads(4)  # Limitar threads para no interferir con GPUs
            
            model = SentenceTransformer(model_name, device=self.device)
            self.models[model_name] = model
            
            logger.info(f"Modelo {model_name} cargado en CPU")
        
        return self.models[model_name]
    
    async def generate_embedding(self, text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
        """Genera embeddings para un texto"""
        model = await self.load_model(model_name)
        
        # Ejecutar en thread pool para no bloquear async
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: model.encode(text, convert_to_tensor=False).tolist()
        )
        
        return embedding

# ============================================================================
# Aplicación FastAPI
# ============================================================================

# Variables globales
gpu_manager = GPUManager()
model_downloader = ModelDownloader()
server_manager = LlamaServerManager()
load_balancer = LoadBalancer(server_manager)
embedding_manager = EmbeddingManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    logger.info("Iniciando sistema Multi-GPU Llama.cpp")
    
    # Configuración pasada por argumentos (se inyecta desde main)
    args = app.state.args
    
    try:
        # 1. Descargar modelo si es necesario
        logger.info(f"Descargando modelo: {args.repo_id}/{args.model_filename}")
        model_path = await model_downloader.download_model(args.repo_id, args.model_filename)
        
        # 2. Detectar GPUs disponibles
        gpu_manager.refresh_gpu_info()
        available_gpus = gpu_manager.get_available_gpus(args.min_vram_gb)
        
        if not available_gpus:
            logger.error("No hay GPUs disponibles con suficiente VRAM")
            sys.exit(1)
        
        logger.info(f"GPUs disponibles: {available_gpus}")
        
        # 3. Iniciar servidores en cada GPU
        tasks = []
        for i, gpu_id in enumerate(available_gpus[:args.max_gpus]):
            config = LlamaServerConfig(
                gpu_id=gpu_id,
                port=8080 + gpu_id,
                model_path=model_path,
                context_size=args.context_size,
                parallel_requests=args.parallel_requests,
                batch_size=args.batch_size,
                max_tokens=args.max_output_tokens
            )
            
            task = server_manager.start_server(config)
            tasks.append(task)
        
        # Esperar a que todos los servidores estén listos
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_servers = sum(1 for r in results if r is True)
        
        if successful_servers == 0:
            logger.error("No se pudo iniciar ningún servidor")
            sys.exit(1)
        
        logger.info(f"✅ {successful_servers} servidores iniciados correctamente")
        
        # 4. Precargar modelo de embeddings
        logger.info(f"Precargando modelo de embeddings: {args.embedding_model}")
        await embedding_manager.load_model(args.embedding_model)
        
        logger.info("Sistema listo para recibir requests")
        
        yield
        
    finally:
        logger.info("Deteniendo sistema...")
        server_manager.stop_all_servers()
        logger.info("Sistema detenido correctamente")

# Crear app FastAPI
app = FastAPI(
    title="Multi-GPU Llama.cpp API",
    description="API distribuida para inferencia con Llama.cpp en múltiples GPUs",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# Endpoints de la API
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check del sistema"""
    servers = server_manager.get_available_servers()
    gpu_manager.refresh_gpu_info()
    
    return {
        "status": "healthy",
        "servers_active": len(servers),
        "gpus_detected": len(gpu_manager.gpus),
        "server_stats": load_balancer.server_stats
    }

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    if request.stream:
        return StreamingResponse(
            stream_completion(request),
            media_type="text/event-stream"
        )
    
    # Use environment-appropriate host
    client_host = getattr(app.state, 'client_host', 'localhost')
    async with LlamaClient(load_balancer, host=client_host) as client:
        result = await client.generate_completion(request)
        
        return CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            created=int(time.time()),
            choices=[{
                "text": result["content"],
                "index": 0,
                "finish_reason": "stop" if result.get("stopped_eos") else "length",
                "metadata": {
                    "gpu_id": result.get("gpu_id"),
                    "processing_time": result.get("processing_time"),
                    "tokens_predicted": result.get("tokens_predicted", 0)
                }
            }],
            usage={
                "prompt_tokens": result.get("tokens_evaluated", 0),
                "completion_tokens": result.get("tokens_predicted", 0),
                "total_tokens": result.get("tokens_evaluated", 0) + result.get("tokens_predicted", 0)
            }
        )

async def stream_completion(request: CompletionRequest):
    """Stream de completion"""
    async with LlamaClient(load_balancer) as client:
        async for chunk in client.generate_completion_stream(request):
            yield chunk
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Genera respuesta de chat"""
    # Convertir mensajes a prompt
    prompt = messages_to_prompt(request.messages)
    
    completion_request = CompletionRequest(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        stop=request.stop,
        stream=request.stream
    )
    
    if request.stream:
        return StreamingResponse(
            stream_completion(completion_request),
            media_type="text/event-stream"
        )
    
    return await create_completion(completion_request)

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convierte mensajes de chat a prompt"""
    prompt_parts = []
    
    for message in messages:
        if message.role == "system":
            prompt_parts.append(f"System: {message.content}")
        elif message.role == "user":
            prompt_parts.append(f"User: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}")
    
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Genera embeddings de texto"""
    try:
        embedding = await embedding_manager.generate_embedding(
            request.input, 
            request.model
        )
        
        return EmbeddingResponse(
            data=[{
                "object": "embedding",
                "index": 0,
                "embedding": embedding
            }],
            usage={
                "prompt_tokens": len(request.input.split()),
                "total_tokens": len(request.input.split())
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando embeddings: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Estadísticas del sistema"""
    gpu_manager.refresh_gpu_info()
    
    return {
        "gpus": [
            {
                "id": gpu.id,
                "name": gpu.name,
                "memory_total_gb": round(gpu.memory_total / 1024, 2),
                "memory_free_gb": round(gpu.memory_free / 1024, 2),
                "memory_used_gb": round((gpu.memory_total - gpu.memory_free) / 1024, 2),
                "utilization": gpu.utilization,
                "temperature": gpu.temperature
            }
            for gpu in gpu_manager.gpus
        ],
        "servers": [
            {
                "gpu_id": gpu_id,
                "port": server_manager.get_server_port(gpu_id),
                "stats": load_balancer.server_stats.get(gpu_id, {})
            }
            for gpu_id in server_manager.get_available_servers()
        ],
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2)
        }
    }

# ============================================================================
# CLI y Main
# ============================================================================

def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Sistema Multi-GPU Llama.cpp con FastAPI"
    )
    
    # Configuración del modelo
    parser.add_argument(
        "--repo-id",
        default="unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",
        help="Repositorio de HuggingFace del modelo"
    )
    parser.add_argument(
        "--model-filename",
        default="Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        help="Nombre del archivo del modelo GGUF"
    )
    
    # Configuración del servidor
    parser.add_argument(
        "--server-binary",
        default="./server",
        help="Ruta al binario del servidor llama.cpp"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=32768,
        help="Tamaño del contexto en tokens (32k por defecto)"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Máximo tokens de salida"
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=2,
        help="Requests paralelos por GPU"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Tamaño del batch"
    )
    
    # Configuración de GPUs
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=8,
        help="Máximo número de GPUs a usar"
    )
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=18.0,
        help="VRAM mínima requerida por GPU (GB)"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        help="IDs específicos de GPUs a usar (ej: 0,1,2)"
    )
    
    # Configuración de embeddings
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Modelo de embeddings para CPU"
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        default=4,
        help="Threads para embeddings en CPU"
    )
    
    # Configuración de la API
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host para la API"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Puerto para la API"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Workers de Uvicorn"
    )
    
    # Configuración de logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging"
    )
    parser.add_argument(
        "--log-file",
        help="Archivo de log (opcional)"
    )
    
    # Configuración de descarga
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Directorio para modelos descargados"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Forzar descarga del modelo aunque exista"
    )
    
    return parser.parse_args()

def setup_logging(args):
    """Configura el sistema de logging"""
    level = getattr(logging, args.log_level)
    
    handlers = [logging.StreamHandler()]
    
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def setup_signal_handlers():
    """Configura manejadores de señales para shutdown graceful"""
    def signal_handler(sig, frame):
        logger.info(f"Recibida señal {sig}, deteniendo servidores...")
        server_manager.stop_all_servers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def validate_environment(args):
    """Valida que el entorno esté correctamente configurado"""
    errors = []
    
    # Verificar binario del servidor
    if not Path(args.server_binary).exists():
        errors.append(f"Binario del servidor no encontrado: {args.server_binary}")
    
    # Verificar nvidia-smi
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        errors.append("nvidia-smi no disponible. ¿Está instalado CUDA?")
    
    # Verificar GPUs disponibles
    gpu_manager.refresh_gpu_info()
    available_gpus = gpu_manager.get_available_gpus(args.min_vram_gb)
    
    if not available_gpus:
        errors.append(f"No hay GPUs con al menos {args.min_vram_gb}GB VRAM disponible")
    
    # Verificar que torch esté disponible para embeddings
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA no disponible para PyTorch, usando solo CPU para embeddings")
    except ImportError:
        errors.append("PyTorch no instalado (requerido para embeddings)")
    
    # Verificar sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        errors.append("sentence-transformers no instalado")
    
    if errors:
        logger.error("Errores de configuración:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True

def print_startup_info(args):
    """Imprime información de inicio del sistema"""
    print("\n" + "="*80)
    print("🚀 SISTEMA MULTI-GPU LLAMA.CPP API")
    print("="*80)
    print(f"📦 Modelo: {args.repo_id}/{args.model_filename}")
    print(f"🖥️  Contexto: {args.context_size:,} tokens")
    print(f"📤 Max output: {args.max_output_tokens:,} tokens")
    print(f"🎯 GPUs máximas: {args.max_gpus}")
    print(f"💾 VRAM mínima: {args.min_vram_gb}GB por GPU")
    print(f"⚡ Requests paralelos: {args.parallel_requests} por GPU")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🧠 Embeddings: {args.embedding_model} (CPU)")
    print(f"🌐 API: http://{args.host}:{args.port}")
    print("="*80)
    
    # Mostrar GPUs detectadas
    gpu_manager.refresh_gpu_info()
    available_gpus = gpu_manager.get_available_gpus(args.min_vram_gb)
    
    print(f"🎮 GPUs detectadas:")
    for gpu in gpu_manager.gpus:
        status = "✅ Disponible" if gpu.id in available_gpus else "❌ Insuficiente VRAM"
        print(f"   GPU {gpu.id}: {gpu.name} - {gpu.memory_free/1024:.1f}GB libre - {status}")
    
    if available_gpus:
        estimated_throughput = len(available_gpus[:args.max_gpus]) * args.parallel_requests
        print(f"⚡ Throughput estimado: ~{estimated_throughput} requests simultáneos")
    
    print("="*80 + "\n")

class HealthCheckServer:
    """Servidor de health check independiente para monitoreo"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.app = FastAPI(title="Health Check")
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.get("/health")
        async def health():
            servers_active = len(server_manager.get_available_servers())
            return {
                "status": "healthy" if servers_active > 0 else "degraded",
                "servers_active": servers_active,
                "timestamp": int(time.time())
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Métricas en formato Prometheus-like"""
            gpu_manager.refresh_gpu_info()
            
            metrics = []
            
            # Métricas de GPU
            for gpu in gpu_manager.gpus:
                metrics.extend([
                    f'gpu_memory_total_bytes{{gpu="{gpu.id}",name="{gpu.name}"}} {gpu.memory_total * 1024 * 1024}',
                    f'gpu_memory_free_bytes{{gpu="{gpu.id}",name="{gpu.name}"}} {gpu.memory_free * 1024 * 1024}',
                    f'gpu_utilization_percent{{gpu="{gpu.id}",name="{gpu.name}"}} {gpu.utilization}',
                    f'gpu_temperature_celsius{{gpu="{gpu.id}",name="{gpu.name}"}} {gpu.temperature}'
                ])
            
            # Métricas de servidores
            for gpu_id in server_manager.get_available_servers():
                stats = load_balancer.server_stats.get(gpu_id, {})
                requests = stats.get("requests", 0)
                avg_time = stats.get("avg_time", 0)
                
                metrics.extend([
                    f'llama_requests_total{{gpu="{gpu_id}"}} {requests}',
                    f'llama_avg_processing_time_seconds{{gpu="{gpu_id}"}} {avg_time}'
                ])
            
            return "\n".join(metrics)

def detect_environment():
    """Detect if running in RunPod or similar container environment"""
    # Check for RunPod-specific environment variables
    if os.environ.get('RUNPOD_POD_ID'):
        return 'runpod'
    # Check for common container indicators
    elif os.path.exists('/.dockerenv'):
        return 'container'
    else:
        return 'local'

async def main():
    """Función principal"""
    args = parse_arguments()

    env_type = detect_environment()
    logger.info(f"Detected environment: {env_type}")
    
    if env_type == 'runpod':
        # RunPod-specific configurations
        args.host = "0.0.0.0"  # Ensure binding to all interfaces
        
        # RunPod typically exposes port 3000 by default
        if args.port == 3000:
            logger.info("Using RunPod default port 3000")
        
        # Disable separate health check server in RunPod
        # as it can cause port conflicts
        logger.info("RunPod environment detected - disabling separate health server")
        run_health_server = False
    else:
        run_health_server = True
    
    # Configurar logging
    setup_logging(args)
    
    # Configurar manejadores de señales
    setup_signal_handlers()
    
    # Mostrar información de inicio
    print_startup_info(args)
    
    # Validar entorno
    if not await validate_environment(args):
        sys.exit(1)
    
    # Configurar directorio de modelos
    global model_downloader
    model_downloader = ModelDownloader(args.models_dir)
    
    # Configurar servidor llama.cpp
    global server_manager
    server_manager = LlamaServerManager(args.server_binary)
    
    # Configurar manager de embeddings
    global embedding_manager
    embedding_manager = EmbeddingManager()
    
    # Filtrar GPUs específicas si se especificaron
    if args.gpu_ids:
        specified_gpus = [int(x.strip()) for x in args.gpu_ids.split(",")]
        available_gpus = gpu_manager.get_available_gpus(args.min_vram_gb)
        available_gpus = [gpu_id for gpu_id in available_gpus if gpu_id in specified_gpus]
        
        if not available_gpus:
            logger.error(f"Ninguna de las GPUs especificadas ({args.gpu_ids}) está disponible")
            sys.exit(1)
        
        logger.info(f"Usando GPUs específicas: {available_gpus}")
    
    global llama_client_host
    llama_client_host = "localhost" if env_type == 'local' else "0.0.0.0"
    
    # CRITICAL FIX: Inject args into app state BEFORE starting the server
    app.state.args = args
    
    # Also set client_host for LlamaClient
    app.state.client_host = llama_client_host
    
    # Start servers conditionally
    try:
        if run_health_server:
            # Start separate health server for local development
            health_server = HealthCheckServer()
            health_config = uvicorn.Config(
                health_server.app,
                host=args.host,
                port=8080,
                log_level=args.log_level.lower(),
                access_log=False
            )
            health_server_instance = uvicorn.Server(health_config)
        
        # Main server configuration
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level.lower(),
            access_log=True,
            reload=False
        )
        
        server = uvicorn.Server(config)
        
        if run_health_server:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(health_server_instance.serve())
                tg.create_task(server.serve())
        else:
            # RunPod: only run main server
            await server.serve()
            
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        sys.exit(1)
    finally:
        logger.info("Limpiando recursos...")
        server_manager.stop_all_servers()

if __name__ == "__main__":
    asyncio.run(main())