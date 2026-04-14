import pandas as pd
import os
from spatial.poi import get_zone_type
from spatial.visualization import create_parking_map

from dotenv import load_dotenv
load_dotenv()

def main():

    data_filename = os.environ.get("PARQUET_SPATIAL_MERGE", "df_spatial_merge.parquet") # geolocated sensor dataset
    data_path = os.path.join("data", data_filename) 
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
    
    # GESTIONAR AQUI LAS FILAS CON NONE (SE HA DEVUELTO 'None' SI HA HABIDO ALGÚN FALLO, REVISAR POI.PY)
    
    # Generate map with classified zones
    print("\nGenerando mapa...")
    parking_map = create_parking_map(df_classified, lat_col=LAT_COL, lon_col=LON_COL, id_col=ID_COL, name_col=NAME_COL)
    
    # Save map in HTML
    output_file = 'map_locations.html'
    parking_map.save(output_file)
    print(f"Mapa guardado en {output_file}")

if __name__ == "__main__":
    main()