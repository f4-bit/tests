from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

class InferenceRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    request_id: str
    generated_text: str
    processing_time: float

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Configura vLLM (ajusta a tu GPU; dtype="auto" para 4bit)
    engine_args = AsyncEngineArgs(
        model="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
        dtype="auto",  # Detecta 4bit
        max_model_len=8192,
        gpu_memory_utilization=0.9,  # Ajusta si OOM
        quantization="bitsandbytes",  # Para bnb-4bit
        max_num_seqs=256,  # Para high throughput
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield
    # Shutdown no necesario, pero puedes agregar

app = FastAPI(lifespan=lifespan)

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    if engine is None:
        raise HTTPException(503, "Modelo no disponible")
    
    request_id = request.request_id or str(uuid.uuid4())
    start_time = time.time()
    
    # Params per-request (vLLM soporta individuales)
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_length,
        # Agrega si quieres: top_p=0.95, etc.
    )
    
    # Genera async
    results = await engine.generate(request.text, sampling_params, request_id)
    
    # Extrae output (vLLM retorna stream, pero aquí sync para simpleza)
    generated_text = results.outputs[0].text if results.outputs else ""
    processing_time = time.time() - start_time
    
    return InferenceResponse(
        request_id=request_id,
        generated_text=generated_text,
        processing_time=processing_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)