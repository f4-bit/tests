from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import uuid
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from collections import defaultdict
import json
from instruction import system_prompt

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    text: str = Field(..., description="Texto de entrada para generar")
    max_length: Optional[int] = Field(100, ge=1, le=2048, description="Máximo tokens a generar")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperatura de sampling")
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling")
    request_id: Optional[str] = Field(None, description="ID único de la petición")
    priority: Optional[int] = Field(0, ge=0, le=10, description="Prioridad (0=baja, 10=alta)")

class BatchInferenceRequest(BaseModel):
    texts: List[str] = Field(..., description="Lista de textos para procesar en batch")
    max_length: Optional[int] = Field(100, ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    batch_id: Optional[str] = Field(None, description="ID del batch")

class InferenceResponse(BaseModel):
    request_id: str
    generated_text: str
    processing_time: float
    input_tokens: int
    output_tokens: int
    batch_size: int = 1  # Nuevo campo para indicar tamaño del batch

class BatchInferenceResponse(BaseModel):
    batch_id: str
    results: List[InferenceResponse]
    total_processing_time: float
    batch_size: int

class BatchingStats(BaseModel):
    active_requests: int
    queued_requests: int
    avg_batch_size: float
    total_processed: int
    avg_processing_time: float

# Variables globales
engine = None
batch_stats = {
    "total_requests": 0,
    "total_batches": 0,
    "batch_size_sum": 0,
    "processing_time_sum": 0.0
}

# Queue para requests con prioridad
request_queue = asyncio.PriorityQueue()
batch_processor_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, batch_processor_task
    logger.info("🚀 Inicializando vLLM engine con batching optimizado...")
    
    try:
        engine_args = AsyncEngineArgs(
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            dtype="auto",
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            #quantization="fp8",
            # Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8

            # CONFIGURACIÓN AVANZADA DE BATCHING
            max_num_seqs=64,  # Reducido para mejor latencia
            max_num_batched_tokens=16384,  # Más tokens por batch
            
            # CHUNKED PREFILL OPTIMIZADO
            enable_chunked_prefill=True,
            
            # OPTIMIZACIONES DE MEMORIA Y SCHEDULING
            enable_prefix_caching=True,
            #use_v2_block_manager=True,
            preemption_mode="recompute", # Deshabilitar prefix caching para qwen3next
            swap_space=4,  # GB de swap space
            
            enforce_eager=False,

            # CONTINUOUS BATCHING
            disable_log_stats=False,  # Para monitorear performance
            trust_remote_code=True,
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Iniciar procesador de batches en background
        batch_processor_task = asyncio.create_task(batch_processor())
        logger.info("✅ Engine y batch processor inicializados")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Error inicializando engine: {e}")
        raise
    finally:
        if batch_processor_task:
            batch_processor_task.cancel()
        logger.info("🛑 Cerrando engine...")

app = FastAPI(
    title="vLLM Advanced Batching API",
    description="API de inferencia con batching dinámico optimizado",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def batch_processor():
    """Procesador de batches en background"""
    logger.info("🔄 Iniciando batch processor...")
    
    while True:
        try:
            # Recopilar requests por un tiempo o hasta llenar batch
            batch_requests = []
            batch_futures = []
            batch_start = time.time()
            
            # Esperar por requests con timeout
            try:
                while True: 
                    priority, (request, future) = await asyncio.wait_for(
                        request_queue.get(), timeout=0.1
                    )
                    batch_requests.append(request)
                    batch_futures.append(future)
            except asyncio.TimeoutError:
                pass
            
            if not batch_requests:
                await asyncio.sleep(0.01)  # 10ms sleep si no hay requests
                continue
            
            # Procesar batch
            logger.info(f"🔥 Procesando batch de {len(batch_requests)} requests")
            results = await process_batch_internal(batch_requests)
            
            # Devolver resultados
            for future, result in zip(batch_futures, results):
                if not future.cancelled():
                    future.set_result(result)
            
            # Actualizar estadísticas
            batch_stats["total_batches"] += 1
            batch_stats["batch_size_sum"] += len(batch_requests)
            
        except Exception as e:
            logger.error(f"❌ Error en batch processor: {e}")
            # Marcar futures como fallidos
            for future in batch_futures:
                if not future.cancelled():
                    future.set_exception(e)
            await asyncio.sleep(0.1)

async def process_batch_internal(requests: List[InferenceRequest]) -> List[InferenceResponse]:
    """Procesar un batch de requests internamente"""
    start_time = time.time()
    
    # Obtener el tokenizer para usar el chat template
    tokenizer = await engine.get_tokenizer()
    
    # Preparar sampling params, request_ids y prompts formateados
    sampling_params_list = []
    request_ids = []
    formatted_prompts = []
    
    for req in requests:
        request_id = req.request_id or str(uuid.uuid4())
        request_ids.append(request_id)
        
        sampling_params = SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_length,
            top_p=req.top_p,
        )
        sampling_params_list.append(sampling_params)
        
        # Formatear el prompt usando el chat template del modelo
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.text}
        ]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Error applying chat template for request {request_id}: {e}")
            # Fallback: formateo manual para Qwen
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{req.text}<|im_end|>\n<|im_start|>assistant\n"
        
        formatted_prompts.append(formatted_prompt)
    
    # Procesar requests de forma concurrente con vLLM AsyncEngine
    tasks = []
    for i, formatted_prompt in enumerate(formatted_prompts):
        # Usar el prompt formateado en lugar del texto crudo
        task = engine.generate(formatted_prompt, sampling_params_list[i], request_ids[i])
        tasks.append(task)
    
    # Función auxiliar para procesar cada generador async
    async def process_single_request(async_gen, req_idx):
        """Procesa un solo request y consume su async_generator"""
        final_output = None
        try:
            async for request_output in async_gen:
                # Cada iteración contiene el estado actualizado de la generación
                final_output = request_output
            return final_output, req_idx
        except Exception as e:
            logger.error(f"Error procesando request {req_idx}: {e}")
            return None, req_idx
    
    # Ejecutar todas las tareas concurrentemente
    process_tasks = [process_single_request(task, i) for i, task in enumerate(tasks)]
    completed_results = await asyncio.gather(*process_tasks, return_exceptions=True)
    
    # Preparar respuestas
    responses = []
    processing_time = time.time() - start_time
    
    for i, (result, req_idx) in enumerate(completed_results):
        req = requests[req_idx]
        
        if isinstance(result, Exception) or result is None:
            # Manejar errores
            logger.error(f"Error en request {req_idx}: {result}")
            generated_text = ""
            output_tokens = 0
        else:
            # Extraer texto generado del resultado final
            generated_text = result.outputs[0].text if result.outputs else ""
            output_tokens = len(result.outputs[0].token_ids) if result.outputs else 0
        
        # Calcular tokens de entrada usando el prompt formateado
        input_tokens = len(tokenizer.encode(formatted_prompts[req_idx]))
        
        response = InferenceResponse(
            request_id=request_ids[req_idx],
            generated_text=generated_text,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            batch_size=len(requests)
        )
        responses.append(response)
    
    # Actualizar estadísticas globales
    batch_stats["total_requests"] += len(requests)
    batch_stats["processing_time_sum"] += processing_time
    
    return responses

@app.get("/stats", response_model=BatchingStats)
async def get_batching_stats():
    """Obtener estadísticas de batching"""
    avg_batch_size = (batch_stats["batch_size_sum"] / batch_stats["total_batches"] 
                     if batch_stats["total_batches"] > 0 else 0)
    avg_processing_time = (batch_stats["processing_time_sum"] / batch_stats["total_batches"] 
                          if batch_stats["total_batches"] > 0 else 0)
    
    return BatchingStats(
        active_requests=0,  # Difícil de calcular en tiempo real
        queued_requests=request_queue.qsize(),
        avg_batch_size=avg_batch_size,
        total_processed=batch_stats["total_requests"],
        avg_processing_time=avg_processing_time
    )

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Generar texto usando batching dinámico"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Crear future para la respuesta
    future = asyncio.Future()
    
    # Agregar a queue con prioridad (prioridad más alta = número más bajo)
    priority = 10 - request.priority  # Invertir para que 10 sea alta prioridad
    await request_queue.put((priority, (request, future)))
    
    try:
        # Esperar resultado del batch processor
        result = await asyncio.wait_for(future, timeout=30.0)  # 30s timeout
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Timeout procesando request")

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(batch_request: BatchInferenceRequest):
    """Generar múltiples textos en un batch explícito"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    batch_id = batch_request.batch_id or str(uuid.uuid4())
    start_time = time.time()
    
    # Convertir a requests individuales
    individual_requests = []
    for text in batch_request.texts:
        req = InferenceRequest(
            text=text,
            max_length=batch_request.max_length,
            temperature=batch_request.temperature,
            top_p=batch_request.top_p
        )
        individual_requests.append(req)
    
    # Procesar batch
    results = await process_batch_internal(individual_requests)
    total_time = time.time() - start_time
    
    return BatchInferenceResponse(
        batch_id=batch_id,
        results=results,
        total_processing_time=total_time,
        batch_size=len(batch_request.texts)
    )

@app.get("/health")
async def health_check():
    """Endpoint de salud con estadísticas de batching"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    stats = await get_batching_stats()
    
    return {
        "status": "healthy",
        "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "batching_enabled": True,
        "queue_size": stats.queued_requests,
        "avg_batch_size": stats.avg_batch_size,
        "total_processed": stats.total_processed
    }

@app.get("/")
async def root():
    return {
        "message": "vLLM Advanced Batching API",
        "features": [
            "Dynamic batching",
            "Priority queuing", 
            "Chunked prefill",
            "Prefix caching",
            "Batch statistics"
        ],
        "endpoints": {
            "generate": "/generate",
            "generate_batch": "/generate_batch", 
            "stats": "/stats",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )