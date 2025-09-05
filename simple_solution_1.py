import os
from llama_cpp import Llama
from llama_cpp.server.app import app as llama_app  # Importa la app directamente
from fastapi import FastAPI
from uvicorn import run
from huggingface_hub import hf_hub_download

# Configura la GPU 0 (RTX 5090)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Descarga y carga el modelo
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
    filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    local_dir="./models"  # Carpeta donde se guardará el modelo
)

# Inicializa el modelo con parámetros específicos
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # Offload todas las capas a la GPU
    n_batch=512,      # Tamaño del batch para procesamiento
    n_ctx=4096,       # Tamaño del contexto
    n_parallel=2,     # Máximo batching de 2 (solicitudes concurrentes)
    verbose=True      # Para depuración
)

# Configura la app FastAPI del servidor de llama-cpp-python
# No pasamos 'llm' a create_app; en su lugar, configuramos el modelo globalmente
os.environ["LLAMA_CPP_MODEL"] = model_path  # Configura el modelo para el servidor
os.environ["LLAMA_CPP_N_PARALLEL"] = "2"    # Batching máximo de 2
os.environ["LLAMA_CPP_N_CTX"] = "4096"      # Tamaño del contexto
os.environ["LLAMA_CPP_N_GPU_LAYERS"] = "-1" # Offload a GPU

# Crea una app FastAPI personalizada
custom_app = FastAPI(title="Servidor LLM con FastAPI")

@custom_app.get("/")
def root():
    return {"message": "Servidor LLM activo. Usa /docs para ver la API o prueba con curl en /v1/chat/completions."}

# Monta el servidor de llama-cpp en la ruta /v1
custom_app.mount("/v1", llama_app)

# Ejecuta el servidor
if __name__ == "__main__":
    run(custom_app, host="0.0.0.0", port=8000)