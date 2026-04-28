#dag.py
from airflow.decorators import dag, task
from datetime import datetime
from typing import Any, List, Dict
import pandas as pd

from src.etl.api import call_endpoint
from src.etl.db_manager import save_to_db

JsonRecord = Dict[str, Any]
JsonPayload = List[JsonRecord]

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
