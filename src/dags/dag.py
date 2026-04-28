#dag.py
from airflow.decorators import dag, task
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.api import call_endpoint
from src.etl.db_manager import save_to_db

JsonRecord = dict[str, Any]
JsonPayload = list[JsonRecord]

@dag(
    dag_id='melbourne_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule='*/15 * * * *', # Trigger every 15 minutes
    catchup=False,
    tags=['TFG', 'API', 'DuckDB']
)
def melbourne_pipeline():

    @task
    def get_api_data() -> JsonPayload:
        
        print("Calling API endpoint...")
        filters = {
            "limit": 100,  
            "offset": 0,
            "order_by": "lastupdated desc" 
        }
        # API call
        data_json = call_endpoint(filters)
        return data_json

    @task
    def df_to_duckdb(data: JsonPayload) -> None:
        if not data:
            print("Data list is empty. No data to save.")
            return
        df = pd.DataFrame(data)
        print(f"{len(df)} registers for DuckDB.")
        rows_total = save_to_db(df, "occupancy")
        print(f"Insertions completed. The database has a total of {rows_total} records.")

    ##############
    data = get_api_data()
    df_to_duckdb(data)


# Pipeline execution
mi_dag_api = melbourne_pipeline()








































# @dag(
#     dag_id='poc_csv_to_duckdb',
#     start_date=datetime(2024, 1, 1),
#     schedule=None, 
#     catchup=False,
#     tags=['TFG', 'PoC']
# )
# def pipeline_prueba():

#     @task
#     def extraer_y_cargar():
#         print("1. Simulando llamada a la API (Leyendo CSV)...")
#         # Leemos el CSV desde la carpeta 'data' del contenedor
#         ruta_csv = '/opt/airflow/data/dummy_api.csv' 
#         df_api = pd.read_csv(ruta_csv)
        
#         print("2. Conectando a DuckDB...")
#         # Guardamos la DB en la carpeta 'duck_db' del contenedor
#         ruta_db = '/opt/airflow/duck_db/tfg_sensores.duckdb'
#         conn = duckdb.connect(ruta_db)
        
#         print("3. Cargando datos en la base de datos...")
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS ocupacion_tiempo_real AS 
#             SELECT * FROM df_api
#         """)
        
#         print("4. Comprobando que los datos están ahí:")
#         resultado = conn.execute("SELECT * FROM ocupacion_tiempo_real").fetchdf()
#         print(resultado)
        
#         conn.close()
#         print("¡Proceso completado con éxito!")

#     tarea_principal = extraer_y_cargar()

# mi_dag = pipeline_prueba()