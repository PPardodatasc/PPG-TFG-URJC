# spatial/main.py
import pandas as pd
import os
from spatial.poi import get_zone_type
from spatial.visualization import create_parking_map

from dotenv import load_dotenv
load_dotenv()

def main():
    # Data path
    data_filename = os.environ.get("PARQUET_SPATIAL_MERGE", "df_spatial_merge.parquet") # geolocated sensor dataset
    data_path = os.path.join("data", data_filename) 
    # Log path
    os.makedirs("logs", exist_ok=True)
    log_filename = os.environ.get("LOG_POIS_NAME", "pois_sensors.log")
    log_path = os.path.join("logs", log_filename)
    # Map path
    os.makedirs("maps", exist_ok=True)
    output_file = os.environ.get("MAP_NAME", "map_locations.html")
    output_path = os.path.join("maps", output_file)

    df = pd.read_parquet(data_path)

    # Variables used
    LAT_COL = 'lat'       
    LON_COL = 'lon'       
    ID_COL = 'DeviceId'
    NAME_COL = 'StreetName'

    # Zone classification based on points of interest from OpenStreetMap
    print(f"Iniciando triangulación espacial para {len(df)} sensores...")
    print("Al tratarse de la API pública de Overpass (OpenStreetMap), el proceso puede tardar unos minutos, especialmente si hay muchos sensores.")
    df_classified = get_zone_type(df, lat_col=LAT_COL, lon_col=LON_COL)
    print("\nResumen de clasificación:")
    print(df_classified['ZoneType'].value_counts())
    
    # Remove sensors with None classification (osm data not available)
    df_classified = df_classified[df_classified['ZoneType'] != 'None']

    # Sensor information log
    with open(log_path, 'w') as f:
        for _, row in df_classified.iterrows():    
            f.write(f"Sensor ID: {row[ID_COL]}, Street: {row[NAME_COL]}, Lat: {row[LAT_COL]}, Lon: {row[LON_COL]}, ZoneType: {row['ZoneType']}\n")
    
    # Generate map with classified zones
    print("\nGenerando mapa...")
    parking_map = create_parking_map(df_classified, lat_col=LAT_COL, lon_col=LON_COL, id_col=ID_COL, name_col=NAME_COL)
    # Save map in HTML
    parking_map.save(output_path)
    print(f"Mapa guardado en {output_path}")

if __name__ == "__main__":
    main()