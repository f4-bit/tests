import asyncio
import subprocess
import uuid
import time
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============= Modelos Pydantic =============
class SingleInferenceRequest(BaseModel):
    text: str = Field(..., description="Texto de entrada para generar")
    max_length: Optional[int] = Field(100, ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    request_id: Optional[str] = Field(None, description="ID único de la petición")


class BatchInferenceRequestAPI(BaseModel):
    texts: List[str] = Field(..., description="Lista de textos para procesar")
    max_length: Optional[int] = Field(100, ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    request_id: str
    generated_text: str
    processing_time: float
    input_tokens: int
    output_tokens: int
    batch_size: int = 1


class UserResponse(BaseModel):
    request_id: str
    status: str
    message: str
    estimated_wait_time: Optional[float] = None


# ============= Dataclasses para manejo interno =============
@dataclass
class QueuedRequest:
    request_id: str
    text: str
    max_length: int
    temperature: float
    top_p: float
    timestamp: float
    future: asyncio.Future = field(default_factory=lambda: asyncio.Future())


# ============= Configuración =============
class Config:
    BACKEND_API_URL = "http://localhost:8000"
    BACKEND_SCRIPT_PATH = "api_vllm.py"  # Ajusta esta ruta a tu script backend
    NUM_WORKERS = 3  # Número de workers concurrentes
    BATCH_SIZE = 6   # Tamaño m de peticiones por worker
    WORKER_POLL_INTERVAL = 0.1  # Segundos entre verificaciones del buffer
    REQUEST_TIMEOUT = 300  # Timeout para requests al backend
    AUTO_START_BACKEND = True  # Flag para auto-inicio del backend
    BACKEND_STARTUP_MAX_WAIT = 600  # 10 minutos para inicio del backend (incluye descarga de modelos)


config = Config()

# Variable global para el proceso del backend
backend_process = None
backend_output_tasks = []  # Para mantener las tareas de lectura de logs


# ============= Buffer/Queue Manager =============
class RequestBuffer:
    def __init__(self):
        self.queue: deque[QueuedRequest] = deque()
        self.lock = asyncio.Lock()
        self.pending_results: Dict[str, asyncio.Future] = {}
    
    async def add_request(self, req: QueuedRequest) -> asyncio.Future:
        """Agrega una petición al buffer"""
        async with self.lock:
            self.queue.append(req)
            self.pending_results[req.request_id] = req.future
        return req.future
    
    async def get_batch(self, batch_size: int) -> List[QueuedRequest]:
        """Obtiene un batch de peticiones del buffer"""
        async with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.queue))):
                if self.queue:
                    batch.append(self.queue.popleft())
            return batch
    
    async def set_result(self, request_id: str, result: InferenceResponse):
        """Establece el resultado de una petición"""
        if request_id in self.pending_results:
            future = self.pending_results[request_id]
            if not future.done():
                future.set_result(result)
            del self.pending_results[request_id]
    
    async def set_error(self, request_id: str, error: Exception):
        """Establece un error para una petición"""
        if request_id in self.pending_results:
            future = self.pending_results[request_id]
            if not future.done():
                future.set_exception(error)
            del self.pending_results[request_id]
    
    def queue_size(self) -> int:
        return len(self.queue)


# ============= Worker =============
class Worker:
    def __init__(self, worker_id: int, buffer: RequestBuffer, batch_size: int):
        self.worker_id = worker_id
        self.buffer = buffer
        self.batch_size = batch_size
        self.is_running = False
        self.processed_batches = 0
    
    async def start(self):
        """Inicia el worker"""
        self.is_running = True
        print(f"Worker {self.worker_id} iniciado")
        
        while self.is_running:
            try:
                # Obtener batch del buffer
                batch = await self.buffer.get_batch(self.batch_size)
                
                if not batch:
                    # No hay peticiones, esperar
                    await asyncio.sleep(config.WORKER_POLL_INTERVAL)
                    continue
                
                # Procesar batch
                await self._process_batch(batch)
                self.processed_batches += 1
                
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: List[QueuedRequest]):
        """Procesa un batch de peticiones"""
        batch_id = f"worker-{self.worker_id}-batch-{self.processed_batches}"
        
        try:
            # Preparar payload para el backend
            payload = {
                "texts": [req.text for req in batch],
                "max_length": batch[0].max_length,
                "temperature": batch[0].temperature,
                "top_p": batch[0].top_p,
                "batch_id": batch_id
            }
            
            print(f"Worker {self.worker_id} procesando {len(batch)} peticiones (batch_id: {batch_id})")
            
            # Hacer request al backend
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{config.BACKEND_API_URL}/generate_batch",
                    json=payload
                )
                response.raise_for_status()
                result_data = response.json()
            
            # Distribuir resultados
            results = result_data.get("results", [])
            for i, req in enumerate(batch):
                if i < len(results):
                    inference_result = InferenceResponse(**results[i])
                    await self.buffer.set_result(req.request_id, inference_result)
                else:
                    await self.buffer.set_error(
                        req.request_id,
                        Exception("No se recibió resultado del backend")
                    )
            
            print(f"Worker {self.worker_id} completó batch {batch_id}")
            
        except Exception as e:
            print(f"Worker {self.worker_id} error procesando batch: {e}")
            # Marcar todas las peticiones del batch como error
            for req in batch:
                await self.buffer.set_error(req.request_id, e)
    
    async def stop(self):
        """Detiene el worker"""
        self.is_running = False
        print(f"Worker {self.worker_id} detenido")


# ============= FastAPI App =============
app = FastAPI(title="API Intermedia con Workers", version="1.0.0")

# Instancias globales
buffer = RequestBuffer()
workers: List[Worker] = []
worker_tasks: List[asyncio.Task] = []


async def check_backend_health():
    """Verifica que el backend esté disponible"""
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{config.BACKEND_API_URL}/")
                if response.status_code == 200:
                    print(f"✓ Backend API disponible en {config.BACKEND_API_URL}")
                    return True
        except Exception as e:
            print(f"⚠ Intento {attempt + 1}/{max_retries}: Backend no disponible - {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
    
    return False


async def stream_process_output(stream, prefix):
    """Lee y muestra el output de un stream en tiempo real"""
    try:
        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None, stream.readline
            )
            if not line:
                break
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            if decoded_line:
                print(f"[{prefix}] {decoded_line}")
    except Exception as e:
        print(f"[{prefix}] Error leyendo stream: {e}")


async def start_backend_api():
    """Inicia la API del backend en puerto 8000"""
    global backend_process, backend_output_tasks
    
    try:
        print(f"🚀 Iniciando backend API desde {config.BACKEND_SCRIPT_PATH}...")
        print("📋 Mostrando logs del backend en tiempo real...")
        print("=" * 80)
        
        backend_process = subprocess.Popen(
            ["python", config.BACKEND_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # Line buffered
            universal_newlines=False
        )
        
        # Crear tareas para leer stdout y stderr en tiempo real
        stdout_task = asyncio.create_task(
            stream_process_output(backend_process.stdout, "BACKEND-OUT")
        )
        stderr_task = asyncio.create_task(
            stream_process_output(backend_process.stderr, "BACKEND-ERR")
        )
        
        # Guardar las tareas para poder cancelarlas después
        backend_output_tasks = [stdout_task, stderr_task]
        
        # Esperar con reintentos más largos para dar tiempo a la descarga del modelo
        print("⏳ Esperando a que el backend inicie (esto puede tomar varios minutos si descarga modelos)...")
        max_wait_time = config.BACKEND_STARTUP_MAX_WAIT
        check_interval = 10  # Verificar cada 10 segundos
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
            
            # Verificar si el proceso murió
            if backend_process.poll() is not None:
                print(f"❌ El proceso del backend terminó inesperadamente (código: {backend_process.returncode})")
                # Dar tiempo a que se impriman los logs finales
                await asyncio.sleep(2)
                return False
            
            # Verificar salud del backend
            print(f"⏳ Verificando backend... ({elapsed_time}s / {max_wait_time}s)")
            if await check_backend_health():
                print("=" * 80)
                print("✅ Backend API iniciada y respondiendo correctamente")
                return True
        
        print("❌ Timeout: El backend no respondió en el tiempo esperado")
        return False
            
    except Exception as e:
        print(f"❌ Error iniciando backend API: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Inicia los workers al arrancar la aplicación"""
    global workers, worker_tasks
    
    # Auto-inicio del backend si está configurado
    if config.AUTO_START_BACKEND:
        print("🔍 Verificando disponibilidad del backend...")
        if not await check_backend_health():
            print("⚠ Backend no disponible. Intentando iniciar...")
            if not await start_backend_api():
                print("⚠ ADVERTENCIA: No se pudo iniciar el backend automáticamente")
                print("Por favor, verifica que el BACKEND_SCRIPT_PATH sea correcto")
    else:
        # Solo verificar
        print("🔍 Verificando disponibilidad del backend...")
        if not await check_backend_health():
            print(f"\n❌ ERROR: No se puede conectar al backend en {config.BACKEND_API_URL}")
            print("Por favor, inicia la API del puerto 8000 manualmente.\n")
    
    print(f"🚀 Iniciando {config.NUM_WORKERS} workers con batch size {config.BATCH_SIZE}")
    
    for i in range(config.NUM_WORKERS):
        worker = Worker(worker_id=i, buffer=buffer, batch_size=config.BATCH_SIZE)
        workers.append(worker)
        task = asyncio.create_task(worker.start())
        worker_tasks.append(task)
    
    print("✅ Workers iniciados correctamente")


@app.on_event("shutdown")
async def shutdown_event():
    """Detiene los workers y el backend al cerrar la aplicación"""
    global backend_process, backend_output_tasks
    
    print("🛑 Deteniendo workers...")
    
    for worker in workers:
        await worker.stop()
    
    # Esperar a que terminen las tareas de workers
    for task in worker_tasks:
        task.cancel()
    
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    # Detener las tareas de lectura de logs del backend
    for task in backend_output_tasks:
        task.cancel()
    
    if backend_output_tasks:
        await asyncio.gather(*backend_output_tasks, return_exceptions=True)
    
    # Detener el backend si lo iniciamos nosotros
    if backend_process:
        print("🛑 Deteniendo backend API...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=10)
            print("✅ Backend detenido correctamente")
        except subprocess.TimeoutExpired:
            print("⚠ Backend no se detuvo a tiempo, forzando terminación...")
            backend_process.kill()
            backend_process.wait()
    
    print("✅ Sistema detenido completamente")


@app.get("/")
async def root():
    return {
        "message": "API Intermedia con Workers",
        "workers": config.NUM_WORKERS,
        "batch_size": config.BATCH_SIZE,
        "queue_size": buffer.queue_size(),
        "backend_url": config.BACKEND_API_URL,
        "auto_start_enabled": config.AUTO_START_BACKEND
    }


@app.get("/stats")
async def get_stats():
    """Obtiene estadísticas del sistema"""
    return {
        "queue_size": buffer.queue_size(),
        "num_workers": config.NUM_WORKERS,
        "batch_size": config.BATCH_SIZE,
        "workers": [
            {
                "worker_id": w.worker_id,
                "is_running": w.is_running,
                "processed_batches": w.processed_batches
            }
            for w in workers
        ],
        "backend_process_running": backend_process is not None and backend_process.poll() is None
    }


@app.post("/inference", response_model=InferenceResponse)
async def inference_single(request: SingleInferenceRequest):
    """Endpoint para una sola petición de inferencia"""
    request_id = request.request_id or str(uuid.uuid4())
    
    # Crear petición en cola
    queued_req = QueuedRequest(
        request_id=request_id,
        text=request.text,
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        timestamp=time.time()
    )
    
    # Agregar al buffer y esperar resultado
    future = await buffer.add_request(queued_req)
    
    try:
        # Esperar resultado (con timeout)
        result = await asyncio.wait_for(future, timeout=config.REQUEST_TIMEOUT)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout esperando respuesta del backend")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando petición: {str(e)}")


@app.post("/inference/batch", response_model=List[InferenceResponse])
async def inference_batch(request: BatchInferenceRequestAPI):
    """Endpoint para múltiples peticiones de inferencia"""
    queued_requests = []
    futures = []
    
    # Agregar todas las peticiones al buffer
    for text in request.texts:
        request_id = str(uuid.uuid4())
        queued_req = QueuedRequest(
            request_id=request_id,
            text=text,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            timestamp=time.time()
        )
        queued_requests.append(queued_req)
        future = await buffer.add_request(queued_req)
        futures.append(future)
    
    try:
        # Esperar todos los resultados
        results = await asyncio.gather(*futures)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando batch: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)