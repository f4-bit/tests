from unsloth import FastLanguageModel
import torch
import asyncio
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from collections import deque
import time
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configuración de optimización para CUDA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Modelos de datos
class InferenceRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    request_id: str
    generated_text: str
    processing_time: float

@dataclass
class QueueItem:
    request_id: str
    text: str
    max_length: int
    temperature: float
    timestamp: float
    future: asyncio.Future

class ModelManager:
    def __init__(self, model_name: str = "unsloth/Qwen3-Coder-30B-A3B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        # Cache para tokenización
        self.token_cache = {}
        
    def load_model(self):
        """Carga el modelo de Unsloth con optimizaciones"""
        if FastLanguageModel is None:
            print("Usando modelo mock para demostración")
            return
            
        try:
            # Optimizaciones adicionales para carga
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=4096,  # Reducido de 8192 para mayor velocidad
                dtype=torch.bfloat16,  # Uso explícito de bfloat16 para mejor rendimiento
                load_in_4bit=True,
                # Configuraciones adicionales para optimización
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Habilitar inferencia rápida con optimizaciones
            FastLanguageModel.for_inference(self.model)
            
            # Compilación del modelo para mayor velocidad (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("Modelo compilado para mayor velocidad")
                except Exception as e:
                    print(f"No se pudo compilar el modelo: {e}")
            
            # Configurar tokenizer con padding token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Modelo {self.model_name} cargado exitosamente")
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            print("Usando modelo mock para demostración")
    
    def prepare_inputs(self, texts: List[str]) -> torch.Tensor:
        """Prepara inputs optimizados"""
        # Usar formato de chat específico para Qwen3-Coder
        formatted_texts = []
        for text in texts:
            # Formato específico para Qwen3-Coder
            formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            formatted_texts.append(formatted_text)
        
        # Tokenización optimizada
        inputs = self.tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # Reducido para mayor velocidad
            return_attention_mask=True,
        ).to(self.device, non_blocking=True)
        
        return inputs
    
    def generate_batch(self, texts: List[str], max_lengths: List[int], 
                      temperatures: List[float]) -> List[str]:
        """Genera respuestas en batch con optimizaciones"""
        if self.model is None or self.tokenizer is None:
            # Mock generation para demostración
            return [f"Respuesta generada para: {text[:50]}..." for text in texts]
        
        try:
            # Preparar inputs optimizados
            inputs = self.prepare_inputs(texts)
            
            # Configuración de generación optimizada
            max_new_tokens = min(max(max_lengths), 512)  # Limitar tokens para velocidad
            avg_temperature = sum(temperatures) / len(temperatures)
            
            # Generar con configuraciones optimizadas
            with torch.no_grad():
                # Usar torch.cuda.amp para mixed precision si está disponible
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=max(0.1, min(avg_temperature, 1.0)),
                            do_sample=True if avg_temperature > 0 else False,
                            top_p=0.9,
                            top_k=50,
                            repetition_penalty=1.1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                            # Optimizaciones adicionales
                            num_beams=1,  # Greedy decoding más rápido
                            early_stopping=True,
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=max(0.1, min(avg_temperature, 1.0)),
                        do_sample=True if avg_temperature > 0 else False,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        early_stopping=True,
                    )
            
            # Decodificación optimizada
            responses = []
            for i, output in enumerate(outputs):
                # Obtener solo los tokens nuevos
                input_length = inputs.input_ids[i].shape[0]
                generated = output[input_length:]
                
                # Decodificación rápida
                response = self.tokenizer.decode(
                    generated, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Limpiar respuesta si contiene tokens de formato
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()
                
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"Error en generación: {e}")
            return [f"Error generando respuesta para: {text[:50]}..." for text in texts]

class BatchProcessor:
    def __init__(self, model_manager: ModelManager, batch_size: int = 8, 
                 max_wait_time: float = 0.1):  # Reducido el tiempo de espera
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.processing = False
        self.lock = threading.Lock()
        self.stats = {
            "total_requests": 0,
            "total_processing_time": 0,
            "avg_processing_time": 0
        }
        
    async def add_request(self, item: QueueItem):
        """Añade un request a la cola"""
        with self.lock:
            self.queue.append(item)
        
        # Iniciar procesamiento si no está corriendo
        if not self.processing:
            asyncio.create_task(self.process_queue())
    
    async def process_queue(self):
        """Procesa la cola en batches con optimizaciones"""
        if self.processing:
            return
            
        self.processing = True
        
        try:
            while True:
                # Esperar hasta tener requests o timeout (más agresivo)
                start_wait = time.time()
                while len(self.queue) == 0 and (time.time() - start_wait) < self.max_wait_time:
                    await asyncio.sleep(0.001)  # Reducido de 0.01 a 0.001
                
                if len(self.queue) == 0:
                    break
                
                # Extraer batch más grande si hay muchos requests
                current_batch_size = min(
                    self.batch_size,
                    len(self.queue),
                    16  # Máximo batch size para no sobrecargar GPU
                )
                
                batch_items = []
                with self.lock:
                    for _ in range(current_batch_size):
                        if self.queue:
                            batch_items.append(self.queue.popleft())
                
                if not batch_items:
                    break
                
                # Procesar batch
                await self.process_batch(batch_items)
                
        finally:
            self.processing = False
    
    async def process_batch(self, batch_items: List[QueueItem]):
        """Procesa un batch de requests con métricas"""
        if not batch_items:
            return
        
        start_time = time.time()
        
        # Extraer datos del batch
        texts = [item.text for item in batch_items]
        max_lengths = [item.max_length for item in batch_items]
        temperatures = [item.temperature for item in batch_items]
        
        # Generar respuestas
        try:
            responses = self.model_manager.generate_batch(texts, max_lengths, temperatures)
            processing_time = time.time() - start_time
            
            # Actualizar estadísticas
            self.stats["total_requests"] += len(batch_items)
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_requests"]
            )
            
            print(f"Batch procesado: {len(batch_items)} requests en {processing_time:.2f}s")
            
            # Enviar respuestas a los futures
            for item, response in zip(batch_items, responses):
                if not item.future.done():
                    result = InferenceResponse(
                        request_id=item.request_id,
                        generated_text=response,
                        processing_time=processing_time
                    )
                    item.future.set_result(result)
                    
        except Exception as e:
            print(f"Error en batch processing: {e}")
            # Manejar errores
            for item in batch_items:
                if not item.future.done():
                    item.future.set_exception(e)

# Variables globales
model_manager = None
batch_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación"""
    global model_manager, batch_processor
    
    # Startup
    print("Iniciando sistema de inferencia optimizado...")
    model_manager = ModelManager()
    model_manager.load_model()
    
    # Configuración optimizada del batch processor
    batch_processor = BatchProcessor(
        model_manager, 
        batch_size=12,  # Incrementado para mejor throughput
        max_wait_time=0.05  # Reducido para menor latencia
    )
    print("Sistema optimizado listo!")
    
    yield
    
    # Shutdown
    print("Cerrando sistema...")
    # Limpiar caché de CUDA si está disponible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Crear aplicación FastAPI
app = FastAPI(
    title="Unsloth Optimized Batch Inference API",
    description="API de inferencia optimizada con batching usando Unsloth",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Endpoint principal para generación de texto optimizada"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Validaciones de entrada
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Texto vacío no permitido")
    
    if request.max_length and request.max_length > 1024:
        request.max_length = 1024  # Limitar para velocidad
    
    # Generar ID único si no se proporciona
    request_id = request.request_id or str(uuid.uuid4())
    
    # Crear future para la respuesta
    future = asyncio.Future()
    
    # Crear item de cola
    queue_item = QueueItem(
        request_id=request_id,
        text=request.text.strip(),
        max_length=request.max_length or 100,
        temperature=max(0.1, min(request.temperature or 0.7, 2.0)),
        timestamp=time.time(),
        future=future
    )
    
    # Añadir a la cola
    await batch_processor.add_request(queue_item)
    
    # Esperar respuesta con timeout reducido
    try:
        result = await asyncio.wait_for(future, timeout=15.0)  # Reducido de 30s
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Timeout en la generación")

@app.get("/health")
async def health_check():
    """Endpoint de salud con métricas adicionales"""
    queue_size = len(batch_processor.queue) if batch_processor else 0
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    
    return {
        "status": "healthy",
        "model_loaded": model_manager is not None and model_manager.model is not None,
        "queue_size": queue_size,
        "processing": batch_processor.processing if batch_processor else False,
        "gpu_memory": gpu_memory,
        "stats": batch_processor.stats if batch_processor else {}
    }

@app.get("/stats")
async def get_stats():
    """Estadísticas del sistema mejoradas"""
    if batch_processor is None:
        return {"error": "Sistema no inicializado"}
    
    return {
        "queue_size": len(batch_processor.queue),
        "batch_size": batch_processor.batch_size,
        "max_wait_time": batch_processor.max_wait_time,
        "is_processing": batch_processor.processing,
        "model_name": model_manager.model_name if model_manager else "Unknown",
        "performance_stats": batch_processor.stats,
        "cuda_available": torch.cuda.is_available(),
        "device": model_manager.device if model_manager else "Unknown"
    }

if __name__ == "__main__":
    # Configuración optimizada para producción
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,  # Un solo worker para compartir el modelo en GPU
        loop="uvloop" if 'uvloop' in globals() else "asyncio"
    )