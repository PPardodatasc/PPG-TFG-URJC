# db_manager.py
import duckdb
import pandas as pd
from pathlib import Path

def save_to_db(df: pd.DataFrame, name_db: str) -> int:
    """
    Guarda un DataFrame de Pandas en la base de datos DuckDB.
    Crea la tabla si no existe y añade los nuevos registros.
    NOTA: la variable df se usa en consultas SQL de DuckDB, se tiene que recibir aunque no se use como variable.
    Devuelve el número total de filas en la tabla histórica.
    """

    def get_db_path(name_db: str) -> str:
        """
        Construye la ruta absoluta hacia la base de datos DuckDB
        """
        # Get project root directory (2 levels above)
        root_dir = Path(__file__).resolve().parent.parent.parent
        db_path = root_dir / 'duck_db' / f"{name_db}.duckdb"
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
