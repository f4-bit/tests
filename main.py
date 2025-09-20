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

# Modelos de datos
class InferenceRequest(BaseModel):
    text: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    request_id: Optional[str] = None
    system_prompt: Optional[str] = None

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
    system_prompt: Optional[str] = None

class ModelManager:
    def __init__(self, model_name: str = "unsloth/Qwen3-Coder-30B-A3B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Carga el modelo de Unsloth"""
        if FastLanguageModel is None:
            print("Usando modelo mock para demostración")
            return
            
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=8192,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Habilitar inferencia rápida
            FastLanguageModel.for_inference(self.model)
            print(f"Modelo {self.model_name} cargado exitosamente")
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            print("Usando modelo mock para demostración")
    
    def format_qwen_prompt(self, user_text: str, system_prompt: Optional[str] = None) -> str:
        """Formatea el prompt usando los tokens nativos de Qwen"""
        if system_prompt is None:
            system_prompt = "Proporciona una respuesta genérica usando como base el contexto proporcionado. No menciones que se hizo una consulta SQL, ni qué hace el SQL, ni menciones las tablas, no te extiendas en explicar."
        
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return formatted_prompt
    
    def generate_batch(self, texts: List[str], max_lengths: List[int], 
                      temperatures: List[float], system_prompts: List[Optional[str]]) -> List[str]:
        """Genera respuestas en batch usando formato nativo de Qwen"""
        if self.model is None or self.tokenizer is None:
            # Mock generation para demostración
            return [f"Respuesta generada para: {text[:50]}..." for text in texts]
        
        try:
            # Formatear todos los prompts usando el formato nativo de Qwen
            formatted_texts = []
            for text, system_prompt in zip(texts, system_prompts):
                formatted_prompt = self.format_qwen_prompt(text, system_prompt)
                formatted_texts.append(formatted_prompt)
            
            # Tokenizar el batch
            inputs = self.tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=8192 - max(max_lengths)  # Reservar espacio para la generación
            ).to(self.device)
            
            # Usar diferentes temperaturas por request si es necesario
            avg_temperature = sum(temperatures) / len(temperatures)
            do_sample = avg_temperature > 0.0
            
            # Generar respuestas
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max(max_lengths),
                    temperature=avg_temperature if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Parámetros adicionales para mejorar calidad
                    top_p=0.9 if do_sample else None,
                    repetition_penalty=1.1,
                    # Asegurar que termine en el token correcto
                    stop_strings=["<|eot_id|>"] if hasattr(self.tokenizer, 'stop_strings') else None
                )
            
            # Decodificar respuestas
            responses = []
            for i, output in enumerate(outputs):
                # Obtener solo los tokens nuevos
                input_length = inputs.input_ids[i].shape[0]
                generated = output[input_length:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True)
                
                # Limpiar la respuesta (remover tokens especiales que puedan quedar)
                response = self.clean_response(response)
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"Error en generación: {e}")
            return [f"Error generando respuesta para: {text[:50]}..." for text in texts]
    
    def clean_response(self, response: str) -> str:
        """Limpia la respuesta removiendo tokens especiales que puedan quedar"""
        # Remover tokens de Qwen que puedan aparecer en la respuesta
        tokens_to_remove = [
            "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
            "<|begin_of_text|>", "system", "user", "assistant"
        ]
        
        for token in tokens_to_remove:
            response = response.replace(token, "")
        
        # Limpiar espacios en blanco extra
        response = response.strip()
        
        return response

class BatchProcessor:
    def __init__(self, model_manager: ModelManager, batch_size: int = 4, 
                 max_wait_time: float = 0.5):
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.processing = False
        self.lock = threading.Lock()
        
    async def add_request(self, item: QueueItem):
        """Añade un request a la cola"""
        with self.lock:
            self.queue.append(item)
        
        # Iniciar procesamiento si no está corriendo
        if not self.processing:
            asyncio.create_task(self.process_queue())
    
    async def process_queue(self):
        """Procesa la cola en batches"""
        if self.processing:
            return
            
        self.processing = True
        
        try:
            while True:
                # Esperar hasta tener requests o timeout
                start_wait = time.time()
                while len(self.queue) == 0 and (time.time() - start_wait) < self.max_wait_time:
                    await asyncio.sleep(0.01)
                
                if len(self.queue) == 0:
                    break
                
                # Extraer batch
                batch_items = []
                with self.lock:
                    for _ in range(min(self.batch_size, len(self.queue))):
                        if self.queue:
                            batch_items.append(self.queue.popleft())
                
                if not batch_items:
                    break
                
                # Procesar batch
                await self.process_batch(batch_items)
                
        finally:
            self.processing = False
    
    async def process_batch(self, batch_items: List[QueueItem]):
        """Procesa un batch de requests"""
        if not batch_items:
            return
        
        start_time = time.time()
        
        # Extraer datos del batch
        texts = [item.text for item in batch_items]
        max_lengths = [item.max_length for item in batch_items]
        temperatures = [item.temperature for item in batch_items]
        system_prompts = [item.system_prompt for item in batch_items]
        
        # Generar respuestas
        try:
            responses = self.model_manager.generate_batch(texts, max_lengths, temperatures, system_prompts)
            processing_time = time.time() - start_time
            
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
    print("Iniciando sistema de inferencia...")
    model_manager = ModelManager()
    model_manager.load_model()
    batch_processor = BatchProcessor(model_manager, batch_size=4, max_wait_time=0.1)
    print("Sistema listo!")
    
    yield
    
    # Shutdown
    print("Cerrando sistema...")

# Crear aplicación FastAPI
app = FastAPI(
    title="Unsloth Batch Inference API with Qwen Native Format",
    description="API de inferencia con batching usando Unsloth y formato nativo de Qwen",
    version="1.1.0",
    lifespan=lifespan
)

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Endpoint principal para generación de texto"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Generar ID único si no se proporciona
    request_id = request.request_id or str(uuid.uuid4())
    
    # Crear future para la respuesta
    future = asyncio.Future()
    
    # Crear item de cola
    queue_item = QueueItem(
        request_id=request_id,
        text=request.text,
        max_length=request.max_length or 512,
        temperature=request.temperature or 0.7,
        timestamp=time.time(),
        future=future,
        system_prompt=request.system_prompt
    )
    
    # Añadir a la cola
    await batch_processor.add_request(queue_item)
    
    # Esperar respuesta con timeout
    try:
        result = await asyncio.wait_for(future, timeout=30.0)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Timeout en la generación")

@app.get("/health")
async def health_check():
    """Endpoint de salud"""
    queue_size = len(batch_processor.queue) if batch_processor else 0
    return {
        "status": "healthy",
        "model_loaded": model_manager is not None and model_manager.model is not None,
        "queue_size": queue_size,
        "processing": batch_processor.processing if batch_processor else False
    }

@app.get("/stats")
async def get_stats():
    """Estadísticas del sistema"""
    if batch_processor is None:
        return {"error": "Sistema no inicializado"}
    
    return {
        "queue_size": len(batch_processor.queue),
        "batch_size": batch_processor.batch_size,
        "max_wait_time": batch_processor.max_wait_time,
        "is_processing": batch_processor.processing,
        "model_name": model_manager.model_name if model_manager else "Unknown"
    }

if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        "main:app",  # Ajusta el nombre del archivo
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1  # Importante: usar solo 1 worker para evitar problemas con CUDA
    )