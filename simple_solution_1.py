import os
from llama_cpp import Llama
from llama_cpp.server.app import create_app
from fastapi import FastAPI
from uvicorn import run
from huggingface_hub import hf_hub_download

# Configura la GPU 0 (RTX 5090)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    filename="Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
    repo_type="model"
)

# Inicializa el modelo con parámetros específicos
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # Offload todas las capas a la GPU
    n_batch=512,      # Tamaño del batch para procesamiento (ajusta si es necesario)
    n_ctx=4096,       # Tamaño del contexto (ajusta según necesidad)
    verbose=True      # Para depuración
)

# Crea la aplicación FastAPI usando el servidor de llama-cpp-python
app = create_app(
    llm=llm,
    settings={
        "n_parallel": 2,  # Máximo batching de 2 (número de solicitudes concurrentes)
        "host": "0.0.0.0",
        "port": 8000
    }
)

# Opcional: Monta endpoints personalizados en una app FastAPI
custom_app = FastAPI(title="Servidor LLM con FastAPI")

@custom_app.get("/")
def root():
    return {"message": "Servidor LLM activo. Usa /docs para ver la API o prueba con curl en /v1/chat/completions."}

# Monta el servidor de llama-cpp en la ruta /v1
custom_app.mount("/v1", app)

# Ejecuta el servidor
if __name__ == "__main__":
    run(custom_app, host="0.0.0.0", port=8000)