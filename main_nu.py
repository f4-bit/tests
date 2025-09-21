import os
# Configurar variables de entorno ANTES de importar cualquier cosa de HF/transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Soluciona el warning de tokenizers
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Opcional: mejor debugging CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Opcional: mejor performance

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_max_length = 8192
        
    def load_model(self):
        """Carga el modelo usando transformers nativo"""
        try:
            print(f"Cargando tokenizer para {self.model_name}...")
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"  # Importante para batching
            )
            
            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Cargando modelo {self.model_name}...")
            
            # Configuración de cuantización 4-bit (similar a unsloth)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",  # Usar Flash Attention si está disponible
                use_cache=True
            )
            
            # Optimizaciones adicionales
            if hasattr(self.model, "eval"):
                self.model.eval()
            
            # Compilar el modelo para mejor performance (opcional)
            if hasattr(torch, "compile") and torch.cuda.is_available():
                try:
                    print("Compilando modelo con torch.compile...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("Modelo compilado exitosamente")
                except Exception as e:
                    print(f"No se pudo compilar el modelo: {e}")
            
            print(f"Modelo {self.model_name} cargado exitosamente")
            print(f"Dispositivo: {next(self.model.parameters()).device}")
            print(f"Tipo de datos: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            print("Usando modelo mock para demostración")
    
    def format_qwen_prompt(self, user_text: str, system_prompt: Optional[str] = None) -> str:
        """Formatea el prompt usando los tokens nativos de Qwen"""
        if system_prompt is None:
            system_prompt = "Proporciona una respuesta genérica usando como base el contexto proporcionado. No menciones que se hizo una consulta SQL, ni qué hace el SQL, ni menciones las tablas, no te extiendas en explicar."
        
        # Usar el formato de chat template de Qwen si está disponible
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                print(f"Error usando chat template: {e}, usando formato manual")
        
        # Formato manual si no hay chat template
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return formatted_prompt
    
    def generate_batch(self, texts: List[str], max_lengths: List[int], 
                      temperatures: List[float], system_prompts: List[Optional[str]]) -> List[str]:
        """Genera respuestas en batch usando transformers nativo"""
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
            max_input_length = self.model_max_length - max(max_lengths) - 50  # Buffer de seguridad
            
            inputs = self.tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_attention_mask=True
            )
            
            # Mover al dispositivo del modelo
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Parámetros de generación
            avg_temperature = sum(temperatures) / len(temperatures)
            do_sample = avg_temperature > 0.0
            max_new_tokens = max(max_lengths)
            
            # Configurar parámetros de generación
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "return_dict_in_generate": False
            }
            
            if do_sample:
                generation_config.update({
                    "temperature": avg_temperature,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.1
                })
            else:
                generation_config.update({
                    "num_beams": 1
                })
            
            # Generar respuestas
            with torch.no_grad():
                # Usar torch.inference_mode para mejor performance
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
            
            # Decodificar respuestas
            responses = []
            for i, output in enumerate(outputs):
                # Obtener solo los tokens nuevos
                input_length = inputs['input_ids'][i].shape[0]
                generated = output[input_length:]
                
                response = self.tokenizer.decode(
                    generated, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Limpiar la respuesta
                response = self.clean_response(response)
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"Error en generación: {e}")
            import traceback
            traceback.print_exc()
            return [f"Error generando respuesta para: {text[:50]}..." for text in texts]
    
    def clean_response(self, response: str) -> str:
        """Limpia la respuesta removiendo tokens especiales que puedan quedar"""
        # Remover tokens de Qwen que puedan aparecer en la respuesta
        tokens_to_remove = [
            "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
            "<|begin_of_text|>", "<|end_of_text|>",
            "system", "user", "assistant"
        ]
        
        cleaned = response
        for token in tokens_to_remove:
            cleaned = cleaned.replace(token, "")
        
        # Limpiar espacios en blanco extra y saltos de línea
        cleaned = cleaned.strip()
        
        # Remover múltiples saltos de línea
        import re
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        return cleaned

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
    print("Iniciando sistema de inferencia con transformers nativo...")
    model_manager = ModelManager()
    model_manager.load_model()
    batch_processor = BatchProcessor(model_manager, batch_size=4, max_wait_time=0.1)
    print("Sistema listo!")
    
    yield
    
    # Shutdown
    print("Cerrando sistema...")
    # Limpiar memoria GPU si es necesario
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Crear aplicación FastAPI
app = FastAPI(
    title="Transformers Native Batch Inference API",
    description="API de inferencia con batching usando transformers nativo y formato nativo de Qwen",
    version="2.0.0",
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
        result = await asyncio.wait_for(future, timeout=45.0)  # Más tiempo para transformers
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Timeout en la generación")

@app.get("/health")
async def health_check():
    """Endpoint de salud"""
    queue_size = len(batch_processor.queue) if batch_processor else 0
    model_loaded = model_manager is not None and model_manager.model is not None
    
    # Información adicional del sistema
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved() / 1024**3     # GB
        }
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "queue_size": queue_size,
        "processing": batch_processor.processing if batch_processor else False,
        "gpu_info": gpu_info
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
        "model_name": model_manager.model_name if model_manager else "Unknown",
        "model_max_length": model_manager.model_max_length if model_manager else "Unknown"
    }

@app.post("/clear_cache")
async def clear_gpu_cache():
    """Limpia la cache de GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return {"message": "GPU cache cleared"}
    return {"message": "GPU not available"}

if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        "main_nu:app",  # Ajusta el nombre del archivo
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1  # Importante: usar solo 1 worker para evitar problemas con CUDA
    )