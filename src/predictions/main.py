# main.py
# usage: python src/predictions/main.py   --model SAITS   --search_space src/predictions/configs_nni/saits_search_space.json   --trials 10
import argparse
import json
import math
import os
import random
import traceback
from predictions.train import Trainer

def sample_parameters(search_space):
    """
    Randomly samples parameters from the given search space.
    The search space can contain:
    - Lists: one value is randomly selected from the list.
    - Log-uniform distributions: a value in log space between min and max is sampled.
    - Fixed values: use the provided value.
    """
    params = {}
    for key, value in search_space.items():
        if isinstance(value, list):
            # Si es una lista ej: "epochs": [20, 40], elegimos uno aleatorio
            params[key] = random.choice(value)
        elif isinstance(value, dict) and value.get("distribution") == "loguniform":
            # Si es loguniform ej: LR
            low = math.log(value["min"])
            high = math.log(value["max"])
            params[key] = math.exp(random.uniform(low, high))
        else:
            # Por si ponemos un valor fijo en el JSON
            params[key] = value
    return params


def save_best_model_summary(logs_path, model_name, best_mae, best_params, best_model_dir):
    """Save a summary of the best model to logs_experiments/best_models_summary.txt"""
    summary_file = os.path.join(logs_path)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"{model_name}:\n")
        f.write(f"Final mae: {best_mae:.5f}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Path: {best_model_dir}\n")
        f.write("="*40 + "\n\n")
    
    

def main():
    parser = argparse.ArgumentParser(description='HPO para PyPOTS')
    parser.add_argument('--model', type=str, required=True, choices=['SAITS', 'CSDI'])
    parser.add_argument('--search_space', type=str, required=True, help='Ruta al archivo JSON')
    parser.add_argument('--trials', type=int, default=10, help='Número de combinaciones a probar')
    args = parser.parse_args()

    # Input data and log paths
    repo_root = os.getenv("PROJECT_ROOT", "")
    parquet_file = os.getenv("PARQUET_MODEL", "df_pypots.parquet")
    log_file = os.getenv("BEST_MODEL_LOG", "best_models_summary.txt")
    data_path = f"{repo_root}/data/{parquet_file}"
    logs_path = f"{repo_root}/logs_experiments/{log_file}"


    print(f"\n INICIANDO BÚSQUEDA HPO PARA: {args.model}")
    with open(args.search_space, 'r') as f:
        search_space = json.load(f)

    best_mae = float('inf')
    best_params = None
    best_model_dir = None

    for trial in range(args.trials):
        params = sample_parameters(search_space)
        
        print("\n" + "="*60)
        print(f"- PRUEBA {trial + 1}/{args.trials} - {args.model}")
        print(f"Parámetros: {params}")
        print("="*60)

        try:
            trainer = Trainer(args.model, repo_root, data_path)
            save_dir, mae = trainer.train(params)
            
            print(f"\n Prueba {trial + 1} COMPLETADA.")
            print(f"Guardada en: {save_dir}")
            print(f"MAE Obtenido: {mae:.5f}\n")
            
            if mae < best_mae:
                best_mae = mae
                best_params = params
                best_model_dir = save_dir
                print(f"¡NUEVO MEJOR MODELO! El MAE bajó a {best_mae:.5f} con los parámetros:\n {best_params}\n")
                
        except Exception as e:
            print(f"\nError en la prueba {trial + 1}: {e}")
            traceback.print_exc()

    print("\n" + "="*15)
    print(f"OPTIMIZACIÓN DE {args.model} COMPLETADA")
    print(f"Mejor MAE Final: {best_mae:.5f}")
    print(f"Mejores Parámetros: {best_params}")
    print(f"Ruta del ganador: {best_model_dir}\n")
    
    if best_model_dir is not None:
        save_best_model_summary(logs_path, args.model, best_mae, best_params, best_model_dir)

    print(f"Mejor configuración y resultados almacenados en: {logs_path}")

    
if __name__ == "__main__":
    main()