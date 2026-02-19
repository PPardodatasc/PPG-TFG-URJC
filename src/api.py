# api.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()

def call_endpoint(filters: dict) -> dict:
    """
    Llama al endpoint con los filtros proporcionados y devuelve los datos en formato JSON.
    """
    
    url = os.environ.get("ENDPOINT_URL")
    if not url:
        print("Error: ENDPOINT_URL environment variable is not set.")
        return {}
    
    try:
        response = requests.get(url, params=filters)
        response.raise_for_status()  # launches an exception for HTTP error codes (400-599). For successful responses(200-299), the code continues to execute normally.
        data = response.json()
        return data

    except requests.exceptions.HTTPError as eh:
        print(f" HTTP Error: {eh}")
        print (f"Server details {response.text}")

    except requests.exceptions.ConnectionError as ec:
        print(f"Internet/URL conexion error: {ec}")
        
    except requests.exceptions.Timeout as et:
        print(f"Timeout error: {et}")
        
    except requests.exceptions.RequestException as e:
        print(f"Unexpected error when calling endpoint: {e}")
        
    return {}