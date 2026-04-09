# utils.py
import requests

def download_from_url(url: str, path: str) -> None:
    """
    Downloads a file from a given URL and saves it to the specified path
    """
    try:
        print(f"Starting download of data.\n It may take a while...\n")
        
        response = requests.get(url, stream=True) # stream=True connects to the URL and allows us to download the content in chunks
        response.raise_for_status() 

        with open(path, "wb") as file:
            # Iterate over chunks of data and write them to the file
            for i, chunk in enumerate(response.iter_content(chunk_size=8192)): 
                if chunk:
                    print(f"Writing chunk {i} to file...")
                    file.write(chunk)
                    
        print(f"Download completed and saved to file in {path}")

    except requests.exceptions.HTTPError as eh:
        print(f"HTTP Error: {eh}")
    except requests.exceptions.ConnectionError as ec:
        print(f"Internet/URL conexion error: {ec}")
    except requests.exceptions.Timeout as et:
        print(f"Timeout error: {et}")
    except requests.exceptions.RequestException as e:
        print(f"Unexpected error when calling endpoint: {e}")