# Análisis mediante Redes de Aprendizaje Profundo de Series Temporales Parcialmente Observadas sobre Datos de Ocupación de Estacionamiento en un Espacio Urbano Inteligente
Este repositorio contiene el código correspondiente al Trabajo de Fin de Grado centrado en la ingesta, limpieza, imputación y predicción de la ocupación de plazas de estacionamiento en la ciudad de Melbourne.

## Requisitos previos

| Herramienta | Versión mínima recomendada |
|---|---|
| Python | 3.12 |
| Docker & Docker Compose | 24.x / 2.x |
| [uv](https://docs.astral.sh/uv/) | última versión estable |
| [Quarto](https://quarto.org/) | 1.4+ (solo para la documentación web) |

## Estructura del repositorio

- **`airflow/`**: directorio generado automáticamente al ejecutar Airflow en su máquina. Utilizado para la configuración, DAGs y logs de Apache Airflow.
- **`dags/`**: contiene el grafo de ejecución de Airflow, que orquesta el proceso continuo de extracción y carga de datos.
- **`data/`**: directorio generado automáticamente al ejecutar el pipeline EDA. Almacena los conjuntos de datos originales y procesados.
- **`docs/`**: archivos .html para la renderización web de Quarto.
- **`duck_db/`**: directorio donde se ubica la base de datos DuckDB local. No se muestra en GitHub por buenas prácticas.
- **`memoria/`**: documentación en LaTeX con los archivos necesarios para compilar el documento PDF final de la memoria del proyecto, así como el archivo PDF ya compilado.
- **`src/`**: código fuente principal del proyecto.
  - `docs/`: archivos de implementación del portal web con Quarto.
  - `eda/`: notebooks para el Análisis Exploratorio de Datos. Se ejecutan en orden: primero el de índice [0] para la descarga de datos, posteriormente el de índice [1].
  - `etl/`: scripts que conforman el proceso ETL en DuckDB, es decir, consultas a la API de datos, almacenamiento en DuckDB y transformación de datos para su posterior uso en la fase de modelado.
  - `spatial/`: scripts encargados de generar el mapa de clasificación espacial de zonas de estacionamiento con Overpass API y OpenStreetMap.
  - `predictions/`: módulos de entrenamiento, inferencia y evaluación para el modelado de PyPOTS.
  - `utils/`: directorio con funciones genéricas que implementan utilidades de propósito general.
- **`logs_experiments/`** y **`models_best/`**: directorios de almacenamiento generados automáticamente al ejecutar el flujo de entrenamiento de los modelos de imputación y predicción. Almacenan los pesos y las configuraciones de los mejores modelos entrenados, los archivos TensorBoard del entrenamiento y las métricas obtenidas en evaluación. Tenga en cuenta que algunos archivos de pesos pueden no estar incluidos debido al límite de tamaño de Git al realizar un push.
- **Archivos `*.sh`**: scripts Bash ejecutables para automatizar la optimización de hiperparámetros de los modelos y su entrenamiento.
- **`Dockerfile`** y **`docker-compose.yaml`**: archivos de configuración para la construcción de imágenes y despliegue de contenedores, fundamentalmente para levantar el orquestador Apache Airflow.

## Instalación y entorno

Es fundamental tener el archivo de configuración `pyproject.toml`. De esta forma, las rutas funcionarán sin importar el directorio de ejecución y Python considerará que `src/` es la raíz del proyecto. Asimismo, es imprescindible crear y disponer de un archivo **`.env`** en el directorio principal. Este archivo debe contener todas las variables de entorno requeridas por la aplicación. Véase el archivo `.env.example`.

### Gestor de paquetes (uv)

Se recomienda utilizar [uv](https://docs.astral.sh/uv/) como gestor de paquetes por su velocidad y compatibilidad con `pip`. Para instalarlo:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Creación del entorno virtual

Antes de instalar ninguna dependencia, cree un entorno virtual de Python en la raíz del proyecto:
```bash
# Crear el entorno virtual
uv venv .venv

# Activar el entorno virtual en Linux/macOS
source .venv/bin/activate

# Activar el entorno virtual en Windows
.venv\Scripts\activate
```

### Instalación de dependencias

Una vez activado el entorno virtual, instale las librerías del proyecto y configure la resolución de rutas:
```bash
# Instalar dependencias
uv pip install -r requirements.txt

# REQUERIDO. Instalar el proyecto en modo editable
uv pip install -e .
```
> **Nota:** si no desea utilizar `uv`, los comandos equivalentes con `pip` son `pip install -r requirements.txt` y `pip install -e .`.


## Portal web de resultados con Quarto

### Versión web compilada
Para explorar la versión web, es posible acceder directamente a la versión desplegada a través de GitHub Pages:

**[Acceder a la web del proyecto](https://PPardodatasc.github.io/PPG-TFG-URJC/)**

### Compilación y ejecución local
Si se desea auditar el código fuente, modificar el contenido o compilar la web de forma local, es necesario disponer de [Quarto](https://quarto.org/) instalado en el sistema.

Para previsualizar la web en tiempo real en un navegador local:
```bash
# Navegar al directorio raíz del portal web desde la raíz del proyecto
cd src/docs 
quarto render
quarto preview
```


## Ejecución de Apache Airflow

Si bien en la implementación del TFG se despliega la arquitectura en un servidor remoto, si desea ejecutar Airflow en su máquina local, se deben ejecutar los siguientes comandos para que todos los archivos se almacenen dentro del repositorio y se mantenga una estructura organizada:

1. **Exportar la variable de entorno HOME**: ejecute el siguiente comando para configurarlo de forma automática y persistente en el entorno virtual (`.venv`):
   ```bash
   echo 'export AIRFLOW_HOME="SU_RUTA_RAÍZ/airflow"' >> .venv/bin/activate
   ```
   *(Basta con ejecutarlo una vez para su entorno virtual, la configuración se mantiene para siempre).*

2. **Inicializar con Docker Compose**:
   ```bash
   # 1. Inicializar metadatos internos de Airflow
   docker compose up airflow-init
   
   # 2. Levantar el servicio de forma que se mantenga en segundo plano
   docker compose up -d
   ```
3. **Acceso a la interfaz**: acceda a `http://localhost:8080` desde su navegador. Se accede con las credenciales configuradas por defecto (normalmente `airflow` como usuario y contraseña).

## Ejecución del pipeline bifuncional: imputación + predicción

Como se indica previamente en la estructura del repositorio, se incluyen scripts en Bash para facilitar la experimentación. Para analizar las curvas de entrenamiento, abra la interfaz de **TensorBoard** en su navegador accediendo a `http://localhost:6006`.

1. **Optimización y entrenamiento de modelos de imputación (SAITS, CSDI)**
   ```bash
   ./run_imputation.sh
   ```
   La mejor configuración de cada modelo se almacena automáticamente en `logs_experiments/best_models_summary.txt` junto a su ruta para poder identificar fácilmente el archivo de pesos y los logs de entrenamiento.

2. **Optimización y entrenamiento de modelos de predicción (DLinear, MICN, Transformer)**
   ```bash
   ./run_forecasting.sh
   ```
   Al igual que en el caso anterior, las configuraciones óptimas se almacenan en `logs_experiments/best_models_summary.txt`.

3. **Ejecución de diferentes experimentos del pipeline secuencial de imputación y predicción**
   ```bash
   ./run_pipeline.sh
   ```
   Este comando ejecutará de manera secuencial los resultados de combinar limpieza y predicción bajo distintos horizontes de tiempo (ej. 24 h, 48 h). El resumen de todas las métricas quedará registrado en `logs_experiments/pipeline_metrics_summary.txt`.


## Visualización del mapa espacial

El módulo `src/spatial/` genera un mapa interactivo en formato HTML con la clasificación espacial de las zonas de estacionamiento. Para visualizarlo, abra el archivo `maps/map_locations.html` directamente en su navegador:
```bash
# Linux
xdg-open maps/map_locations.html

# macOS
open maps/map_locations.html

# Windows
start maps/map_locations.html
```
Alternativamente, puede hacer doble clic sobre el archivo desde el explorador de archivos de su sistema operativo.


## Licencia

Este proyecto se distribuye bajo la licencia **GNU General Public License v3.0 (GPLv3)**. Consulte el archivo [LICENSE](LICENSE) para más detalles.

## Autor

Proyecto desarrollado por Pablo Pardo Gutiérrez como Trabajo de Fin de Grado en la Universidad Rey Juan Carlos.