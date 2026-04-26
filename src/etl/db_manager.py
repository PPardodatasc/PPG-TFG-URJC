# db_manager.py
import duckdb
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

def save_to_db(df: pd.DataFrame, name_db: str) -> int:
    """
    Saves a Pandas DataFrame to a DuckDB database.
    Creates the table if it does not exist and appends new records.
    NOTE: the df variable is used in DuckDB SQL queries, so it must be passed even if not used as a regular Python variable.
    Returns the total number of rows in the historical table.
    """

    def get_db_path(name_db: str) -> str:
        """
        Builds the absolute path to the DuckDB database.
        Uses a fixed path when running in Docker, or computes the local path otherwise.
        """
        # 
        db_path = Path(os.environ.get("DUCKDB_PATH")) / f"{name_db}.duckdb"
        # else: # FALLBACK LOCAL
        #     # Comportamiento original para cuando ejecutas en local
        #     root_dir = Path(__file__).resolve().parent.parent.parent
        #     db_path = root_dir / 'duck_db' / f"{name_db}.duckdb"
            
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return str(db_path)
    
    path_db = get_db_path(name_db)
    print(f"Connecting to DuckDB in {path_db}")
    conn = duckdb.connect(path_db)

    try:
        print("Saving data (Safe mode to avoid duplicates)...")
        # Creates the table if it does not exist (only 1st time)
        # DuckDB treats the df variable as a SQL table when it is a pandas dataframe. 
        # Since 1 is never equal to 0 (WHERE 1=0), this condition is false. This makes DuckDB copy only the structure of the dataframe, without copying any data itself
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {name_db} AS 
            SELECT * FROM df WHERE 1=0
        """)
        
        # Insert data into the previously created table
        conn.execute(f"INSERT INTO {name_db} SELECT * FROM df")
        
        # Verification: count total rows in the table after insertion
        rows = conn.execute(f"SELECT COUNT(*) FROM {name_db}").fetchone()[0] 
        
        return rows
        
    except Exception as e:
        print(f"Error while saving to DuckDB: {e}")
        raise e
        
    finally:
        # This ensures that the connection ALWAYS closes, even if there is an error
        conn.close()
