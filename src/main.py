#main.py
from api import call_endpoint

"""
(Nota: Las APIs suelen poner un límite máximo total. En el caso de Melbourne, 
si intentas pedir más de 10.000 registros usando limit+offset, te dará error. 
Para descargas masivas, las APIs modernas tienen otra ruta que termina en /exports en lugar de /records).
"""




#############
filters = {
    "limit": 10, 
    "select": "kerbsideid, lastupdated, status_description", 
    "order_by": "lastupdated desc" 
}
data=call_endpoint(filters)
print(data)
print(data['results'][0])
