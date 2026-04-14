# utils.py
import requests
import pandas as pd

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



def find_coordinates(row, df_spatial):
    """
    Match the street and transversal streets from the sensor data with the road segment descriptions 
    in the spatial dataset to find the corresponding latitude and longitude.
    """
    # Clean street names
    street = str(row['StreetName']).lower().strip() if pd.notna(row['StreetName']) else ""
    b1 = str(row['BetweenStreet1']).lower().strip() if pd.notna(row['BetweenStreet1']) else ""
    b2 = str(row['BetweenStreet2']).lower().strip() if pd.notna(row['BetweenStreet2']) else ""
        
    # Possible patterns to match in the spatial dataset
    pat1 = f"{street} between {b1} and {b2}"
    pat2 = f"{street} between {b2} and {b1}"
    
    # Search for matches in the spatial dataset 
    match = df_spatial[
        df_spatial['desc_lower'].str.contains(pat1, regex=False) |
        df_spatial['desc_lower'].str.contains(pat2, regex=False)
    ]
    
    if not match.empty:
        # If match, return latitude and longitude. Constant latitude and longitude for all sensors in the same segment.
        return pd.Series([match.iloc[0]['latitude'], match.iloc[0]['longitude']])
    
    return pd.Series([None, None])