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
                            utilization=int(values[4]) if values[4] != '[Not Supported]' else 0,
                            temperature=int(values[5]) if values[5] != '[Not Supported]' else 0
                        ))
                        
                logger.info(f"Detectadas {len(self.gpus)} GPUs")
                for gpu in self.gpus:
                    logger.info(f"  GPU {gpu.id}: {gpu.name} - {gpu.memory_free}/{gpu.memory_total} MB libre")
            else:
                logger.warning(f"nvidia-smi falló con código {result.returncode}: {result.stderr}")
                # Fallback: assume at least GPU 0 exists if CUDA is available
                self._create_fallback_gpu()
                        
        except FileNotFoundError:
            logger.warning("nvidia-smi no encontrado")
            self._create_fallback_gpu()
        except Exception as e:
            logger.error(f"Error detectando GPUs: {e}")
            self._create_fallback_gpu()
    
    def _create_fallback_gpu(self):
        """Crea una GPU fallback para entornos donde nvidia-smi no funciona"""
        try:
            # Check if CUDA is available through torch
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Fallback: detectadas {gpu_count} GPUs via PyTorch")
                
                self.gpus = []
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    # Convert bytes to MB
                    total_memory = props.total_memory // (1024 * 1024)
                    # Assume 80% of memory is free (rough estimate)
                    free_memory = int(total_memory * 0.8)
                    
                    self.gpus.append(GPUInfo(
                        id=i,
                        name=props.name,
                        memory_total=total_memory,
                        memory_free=free_memory,
                        utilization=0,  # Unknown
                        temperature=0   # Unknown
                    ))
                    logger.info(f"  Fallback GPU {i}: {props.name} - ~{free_memory}MB libre (estimado)")
            else:
                logger.warning("CUDA no disponible")
        except ImportError:
            logger.warning("PyTorch no disponible para detectar GPUs")
        except Exception as e:
            logger.error(f"Error en fallback GPU detection: {e}")
    
    def get_available_gpus(self, min_memory_gb: float = 18.0) -> List[int]:
        """Obtiene GPUs con suficiente memoria libre"""
        if not self.gpus:
            logger.warning("No hay GPUs detectadas, intentando GPU 0 como fallback")
            return [0]  # Assume GPU 0 exists as last resort
            
        min_memory_mb = min_memory_gb * 1024
        available = [
            gpu.id for gpu in self.gpus 
            if gpu.memory_free >= min_memory_mb
        ]
        
        # If no GPUs meet memory requirements but we have GPUs, try anyway
        if not available and self.gpus:
            logger.warning(f"Ninguna GPU cumple {min_memory_gb}GB, usando todas disponibles")
            available = [gpu.id for gpu in self.gpus]
        
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
        
        # Verify server binary exists and is executable
        if not self._verify_server_binary():
            return False
            
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
            #"--log-format", "json",
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
            
            # Check if process started successfully
            await asyncio.sleep(2)  # Give it a moment to start
            
            if process.poll() is not None:
                # Process has already exited
                stdout, stderr = process.communicate()
                logger.error(f"Servidor GPU {gpu_id} falló al iniciar:")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                
                # Clean up
                if gpu_id in self.servers:
                    del self.servers[gpu_id]
                if gpu_id in self.configs:
                    del self.configs[gpu_id]
                    
                return False
            
            # Wait for server to be ready with more robust checking
            await self._wait_for_server_improved(config.port, gpu_id)
            logger.info(f"Servidor GPU {gpu_id} listo en puerto {config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando servidor GPU {gpu_id}: {e}")
            
            # Clean up if process was created
            if gpu_id in self.servers:
                process = self.servers[gpu_id]
                if process.poll() is None:  # Still running
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                
                del self.servers[gpu_id]
                if gpu_id in self.configs:
                    del self.configs[gpu_id]
            
            return False
    
    def _verify_server_binary(self) -> bool:
        """Verifica que el binario del servidor existe y es ejecutable"""
        binary_path = Path(self.server_binary)
        
        if not binary_path.exists():
            logger.error(f"Binario del servidor no encontrado: {self.server_binary}")
            logger.info("Sugerencias:")
            logger.info("1. Compila llama.cpp: cmake --build . --target server")
            logger.info("2. Verifica la ruta del binario")
            logger.info("3. Usa --server-binary para especificar ruta correcta")
            return False
        
        if not os.access(binary_path, os.X_OK):
            logger.error(f"Binario del servidor no es ejecutable: {self.server_binary}")
            logger.info(f"Ejecuta: chmod +x {self.server_binary}")
            return False
        
        return True
    
    async def _wait_for_server_improved(self, port: int, gpu_id: int, timeout: int = 90):
        """Espera a que el servidor esté listo con mejor manejo de errores"""
        start_time = time.time()
        last_error = None
        
        # Try different endpoints that llama.cpp server might expose
        test_endpoints = [
            "/health",
            "/v1/models", 
            "/props",
            "/",
        ]
        
        while time.time() - start_time < timeout:
            process = self.servers.get(gpu_id)
            if process and process.poll() is not None:
                # Process has died
                stdout, stderr = process.communicate()
                logger.error(f"Proceso del servidor GPU {gpu_id} murió durante inicialización:")
                logger.error(f"STDOUT: {stdout[-1000:]}")  # Last 1000 chars
                logger.error(f"STDERR: {stderr[-1000:]}")
                raise RuntimeError(f"Proceso del servidor GPU {gpu_id} falló")
            
            # Try to connect to any of the endpoints
            for endpoint in test_endpoints:
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        response = await client.get(f"http://localhost:{port}{endpoint}")
                        if response.status_code in [200, 404]:  # 404 is OK too, means server is responding
                            logger.info(f"Servidor GPU {gpu_id} respondiendo en {endpoint}")
                            return
                except Exception as e:
                    last_error = e
                    continue
            
            await asyncio.sleep(2)
        
        # If we get here, timeout occurred
        process = self.servers.get(gpu_id)
        if process:
            stdout, stderr = process.communicate() if process.poll() is not None else ("", "")
            logger.error(f"Timeout esperando servidor GPU {gpu_id}. Último error: {last_error}")
            if stdout:
                logger.error(f"STDOUT: {stdout[-1000:]}")
            if stderr:
                logger.error(f"STDERR: {stderr[-1000:]}")
        
        raise TimeoutError(f"Servidor en puerto {port} no respondió en {timeout}s")
    
    async def _wait_for_server(self, port: int, timeout: int = 60):
        """Método original mantenido por compatibilidad"""
        await self._wait_for_server_improved(port, 0, timeout)
    
    def stop_all_servers(self):
        """Detiene todos los servidores"""
        for gpu_id, process in self.servers.items():
            logger.info(f"Deteniendo servidor GPU {gpu_id}")
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Forzando terminación del servidor GPU {gpu_id}")
                    process.kill()
                    process.wait()
        
        self.servers.clear()
        self.configs.clear()
    
    def get_available_servers(self) -> List[int]:
        """Obtiene lista de servidores disponibles"""
        # Filter out dead processes
        active_servers = []
        dead_servers = []
        
        for gpu_id, process in self.servers.items():
            if process.poll() is None:  # Still running
                active_servers.append(gpu_id)
            else:
                dead_servers.append(gpu_id)
        
        # Clean up dead servers
        for gpu_id in dead_servers:
            logger.warning(f"Servidor GPU {gpu_id} murió, removiendo")
            del self.servers[gpu_id]
            if gpu_id in self.configs:
                del self.configs[gpu_id]
        
        return active_servers
    
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
    warnings = []
    
    # Verificar binario del servidor
    server_path = Path(args.server_binary)
    if not server_path.exists():
        errors.append(f"Binario del servidor no encontrado: {args.server_binary}")
        logger.error("Sugerencias para resolver problema del servidor:")
        logger.error("1. Clona llama.cpp: git clone https://github.com/ggerganov/llama.cpp.git")
        logger.error("2. Compila: cd llama.cpp && make server")
        logger.error("3. Usa --server-binary para especificar ruta correcta")
    elif not os.access(server_path, os.X_OK):
        errors.append(f"Binario del servidor no es ejecutable: {args.server_binary}")
        logger.error(f"Ejecuta: chmod +x {args.server_binary}")
    else:
        logger.info(f"✅ Binario del servidor encontrado: {args.server_binary}")
    
    # Verificar nvidia-smi (no crítico)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=10)
        logger.info("✅ nvidia-smi disponible")
    except subprocess.TimeoutExpired:
        warnings.append("nvidia-smi timeout - GPU monitoring limitado")
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.append("nvidia-smi no disponible - usando detección CUDA alternativa")
    
    # Verificar CUDA con PyTorch
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA disponible - {gpu_count} GPU(s) detectadas")
            
            # Log GPU details
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"   GPU {i}: {props.name} - {memory_gb:.1f}GB VRAM")
        else:
            errors.append("CUDA no está disponible en PyTorch")
    except ImportError:
        errors.append("PyTorch no instalado (requerido para detección CUDA y embeddings)")
    
    # Verificar GPUs disponibles usando nuestro manager
    gpu_manager.refresh_gpu_info()
    available_gpus = gpu_manager.get_available_gpus(args.min_vram_gb)
    
    if not available_gpus and not cuda_available:
        errors.append(f"No hay GPUs detectadas o disponibles")
    elif not available_gpus and cuda_available:
        warnings.append(f"GPUs detectadas pero ninguna con {args.min_vram_gb}GB VRAM libre")
        logger.warning("Continuando con todas las GPUs disponibles...")
    else:
        logger.info(f"✅ {len(available_gpus)} GPU(s) disponibles con suficiente VRAM")
    
    # Verificar sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✅ sentence-transformers disponible")
        
        # Test load a small model to verify it works
        try:
            test_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info("✅ Embeddings funcionando correctamente")
        except Exception as e:
            warnings.append(f"Error cargando modelo de embeddings de prueba: {e}")
            
    except ImportError:
        errors.append("sentence-transformers no instalado (pip install sentence-transformers)")
    
    # Verificar otras dependencias críticas
    required_modules = [
        ('aiohttp', 'aiohttp'),
        ('aiofiles', 'aiofiles'), 
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('httpx', 'httpx'),
        ('huggingface_hub', 'huggingface-hub'),
        ('psutil', 'psutil')
    ]
    
    missing_modules = []
    for module_name, pip_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(pip_name)
    
    if missing_modules:
        errors.append(f"Módulos faltantes: {', '.join(missing_modules)}")
        logger.error(f"Instala con: pip install {' '.join(missing_modules)}")
    
    # Verificar espacio en disco para modelos
    try:
        models_path = Path(args.models_dir)
        models_path.mkdir(exist_ok=True)
        
        # Check available space (rough estimate - 50GB should be enough for most models)
        statvfs = os.statvfs(models_path)
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        if free_space_gb < 10:
            warnings.append(f"Poco espacio libre para modelos: {free_space_gb:.1f}GB")
        else:
            logger.info(f"✅ Espacio disponible para modelos: {free_space_gb:.1f}GB")
            
    except Exception as e:
        warnings.append(f"No se pudo verificar espacio en disco: {e}")
    
    # Verificar conectividad a Hugging Face (no crítico)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://huggingface.co")
            if response.status_code == 200:
                logger.info("✅ Conectividad a Hugging Face OK")
            else:
                warnings.append("Problemas de conectividad a Hugging Face")
    except Exception:
        warnings.append("No se pudo verificar conectividad a Hugging Face")
    
    # Mostrar warnings
    if warnings:
        logger.warning("⚠️  Advertencias de configuración:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
        logger.warning("El sistema puede funcionar con limitaciones")
    
    # Mostrar errores
    if errors:
        logger.error("❌ Errores de configuración críticos:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("✅ Validación del entorno completada exitosamente")
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