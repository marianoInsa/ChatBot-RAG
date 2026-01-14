import requests
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Tiempo de ejecución: {elapsed:.4f} segundos")
        return result
    return wrapper

@measure_time
def hacer_request():
    response = requests.post(
        "http://localhost:8000/rag/invoke",
        json={
            'input': "Hola! Qué es Hermanos Jota?"
        }
    )
    print("Respuesta del servidor RAG: ", response.json()['output'])

hacer_request()