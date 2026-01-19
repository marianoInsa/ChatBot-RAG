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
            "input": {
                "question": "Hola! ¿Qué es Hermanos Jota?"
            }
        },
        timeout=30
    )

    print("Status code:", response.status_code)
    response.raise_for_status()
    print("Respuesta del servidor:", response.json()["output"]["response"])

hacer_request()