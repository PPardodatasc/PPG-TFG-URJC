# api.py
import os
import requests
from dotenv import load_dotenv
import time
load_dotenv()


def call_endpoint(filters: dict) -> list:
    """
    Llama al endpoint con los filtros proporcionados y devuelve los datos en formato JSON.
    """
    
    url = os.environ.get("ENDPOINT_URL")
    if not url:
        print("Error: ENDPOINT_URL environment variable is not set.")
        return []
    
    records = []
    while True:

        try:
            response = requests.get(url, params=filters)
            response.raise_for_status()  # launches an exception for HTTP error codes (400-599). For successful responses(200-299), the code continues to execute normally.
            print(f"Retrieving data with offset {filters['offset']}...")

            data = response.json()
            data_call=data.get("results", [])

            # If the API returns an empty list of results, break the loop
            if not data_call:
                print("No more records to fetch...")
                break
            records.extend(data_call)

            filters["offset"] += filters["limit"] 

            time.sleep(0.5)

        except requests.exceptions.HTTPError as eh:
            print(f" HTTP Error: {eh}")
            print (f"Server details {response.text}")

        except requests.exceptions.ConnectionError as ec:
            print(f"Internet/URL conexion error: {ec}")
            
        except requests.exceptions.Timeout as et:
            print(f"Timeout error: {et}")
            
        except requests.exceptions.RequestException as e:
            print(f"Unexpected error when calling endpoint: {e}")

    print(f"Extraction completed! Sample length: {len(records)}")    
    return records