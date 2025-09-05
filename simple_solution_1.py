import os
from llama_cpp.server import app
from fastapi import FastAPI
from uvicorn import run

# Configura la GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configura los parámetros del modelo y servidor directamente
# (Esto simula los argumentos de línea de comandos en el código)
from llama_cpp.server.settings import settings

settings.model = "path/to/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"  # Reemplaza con la ruta real al archivo GGUF
settings.n_gpu_layers = -1  # Offload todas las capas a la GPU para máxima eficiencia
settings.n_parallel = 2  # Máximo batching/concurencia de 2 (número máximo de slots paralelos para generaciones simultáneas)
settings.n_batch = 512  # Tamaño de batch para procesamiento de prompts (ajusta si es necesario, pero 512 es un valor estándar eficiente)
settings.port = 8000  # Puerto para el servidor
settings.host = "0.0.0.0"  # Escucha en todas las interfaces

# El servidor de llama-cpp-python ya usa FastAPI internamente y expone endpoints compatibles con OpenAI API,
# como /v1/chat/completions para inferencia.

# Para personalizar más (opcional), puedes montar la app en una FastAPI personalizada si quieres agregar endpoints extras.
custom_app = FastAPI(title="Servidor LLM con FastAPI")

@custom_app.get("/")
def root():
    return {"message": "Servidor LLM activo. Usa /docs para ver la API o prueba con curl en /v1/chat/completions."}

# Monta la app de llama-cpp-server en tu app personalizada
custom_app.mount("/v1", app)

# Ejecuta el servidor
if __name__ == "__main__":
    run(custom_app, host=settings.host, port=settings.port)