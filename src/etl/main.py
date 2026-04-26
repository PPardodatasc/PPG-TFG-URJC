#main.py
from api import call_endpoint
import pandas as pd
"""
(Nota: Las APIs suelen poner un límite máximo total. En el caso de Melbourne, 
si intentas pedir más de 10.000 registros usando limit+offset, te dará error. 
Para descargas masivas, las APIs modernas tienen otra ruta que termina en /exports en lugar de /records).
"""

#############
filters = {
    "limit": 100, 
    "offset": 0,
    #"select": "kerbsideid, lastupdated, status_description", 
    "order_by": "lastupdated desc" 
}
results=call_endpoint(filters)

if results:
        # Lo pasamos a Pandas para verlo bonito y analizar las columnas
        df_realtime = pd.DataFrame(results)
        
        print("\n--- COLUMNAS DISPONIBLES EN TIEMPO REAL ---")
        df_realtime.info()
        
        print("\n--- VISTA PREVIA ---")
        print(df_realtime.head(3))