import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from typing import Any
from predictions.data_loader import DataLoader
from predictions.model_configs import DEFAULT_MODEL_KWARGS
from pypots.imputation import SAITS, CSDI
from pypots.forecasting.dlinear import DLinear
from pypots.forecasting.transformer import Transformer
from pypots.forecasting.micn import MICN

class PipelineEvaluator:
    def __init__(self, imputer_name: str, forecaster_name: str, pred_steps: int):
        self.imputer_name = imputer_name
        self.forecaster_name = forecaster_name
        self.pred_steps = pred_steps
        
        forecaster_config = DEFAULT_MODEL_KWARGS.get(self.forecaster_name) # work with forecaster window_size
        if not forecaster_config:
            raise ValueError(f"No se encontró configuración para {self.forecaster_name} en model_configs.py")
        self.window_size = forecaster_config['window_size']
        
        self.repo_root = os.environ.get("PROJECT_ROOT", os.getcwd())
        self.summary_file = os.path.join(self.repo_root, "logs_experiments", "best_models_summary.txt")
        self.data_path = os.path.join(self.repo_root, "data", os.environ.get("PARQUET_MODEL", "df_pypots.parquet"))
        
        os.environ["TASK_TYPE"] = "forecasting"
        print(f"Iniciando Pipeline: {imputer_name} -> {forecaster_name}")
        print(f"Dimensión maestra dictada por el Forecaster: {self.window_size} pasos")
        
    def _get_path_from_summary(self, target_model: str) -> str:
        if not os.path.exists(self.summary_file):
            raise FileNotFoundError(f"No se encuentra el archivo de resumen en: {self.summary_file}")
            
        model_path = ""
        with open(self.summary_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{target_model}:"):
                found = True
                for j in range(1, 7):
                    if i+j >= len(lines): break
                    if "Path:" in lines[i+j]:
                        base_dir = lines[i+j].split("Path: ")[1].strip()
                        if not base_dir.endswith(".pypots"):
                            base_dir = os.path.join(base_dir, f"{target_model.split('_')[0]}.pypots")
                        model_path = base_dir
                break
                
        if not found or not model_path:
            raise ValueError(f"No se encontró la ruta para {target_model} en el resumen.")
            
        return model_path

    def _load_imputer(self) -> Any:
        config = DEFAULT_MODEL_KWARGS.get(self.imputer_name)
        if not config:
            raise ValueError(f"No se encontró configuración para {self.imputer_name}")
            
        path = self._get_path_from_summary(self.imputer_name)
        print(f"Cargando Imputador {self.imputer_name} desde: {path}")
        
        # Imputers use original window size, not the forecasting model's one
        if self.imputer_name.startswith("SAITS"):
            model = SAITS(
                n_steps=config['window_size'], 
                n_features=9, 
                n_layers=config['n_layers'],
                d_model=config['d_model'], 
                d_ffn=config.get('d_ffn', config['d_model']),
                n_heads=config['n_heads'], 
                d_k=config['d_model'] // config['n_heads'],
                d_v=config['d_model'] // config['n_heads'], 
                dropout=config['dropout'],
                epochs=1 
            )
        elif self.imputer_name.startswith("CSDI"):
            model = CSDI(
                n_steps=config['window_size'], 
                n_features=9, 
                n_layers=config['n_layers'],
                n_channels=config.get('n_channels', 64), 
                n_heads=config['n_heads'], 
                target_strategy=config.get('target_strategy', 'random'),
                n_diffusion_steps=config.get('n_diffusion_steps', 30),
                d_time_embedding=config.get('d_time_embedding', 128),
                d_feature_embedding=config.get('d_feature_embedding', 128),
                d_diffusion_embedding=config.get('d_diffusion_embedding', 128),
                epochs=1
            )
        else:
            raise ValueError("Imputador no soportado.")
            
        model.load(path)
        return model

    def _load_forecaster(self) -> Any:
        config = dict(DEFAULT_MODEL_KWARGS.get(self.forecaster_name, {}))
        path = self._get_path_from_summary(self.forecaster_name)
        print(f"Cargando Forecaster {self.forecaster_name} desde: {path}")
        
        base_name = self.forecaster_name.split("_")[0]
        
        if base_name == "DLinear":
            model = DLinear(
                n_steps=self.window_size, n_features=9, 
                n_pred_steps=self.pred_steps, n_pred_features=9,
                moving_avg_window_size=config.get('moving_avg_window_size', 25),
                d_model=config.get('d_model', 128), 
                epochs=1
            )
        elif base_name == "Transformer":
            model = Transformer(
                n_steps=self.window_size, n_features=9, 
                n_pred_steps=self.pred_steps, n_pred_features=9,
                n_encoder_layers=config.get('n_encoder_layers', config.get('n_layers', 2)),
                n_decoder_layers=config.get('n_decoder_layers', config.get('n_layers', 2)), 
                d_model=config['d_model'], 
                d_ffn=config.get('d_ffn', config['d_model'] * 2), 
                n_heads=config['n_heads'], 
                d_k=config['d_model'] // config['n_heads'],
                d_v=config['d_model'] // config['n_heads'], 
                dropout=config['dropout'], 
                epochs=1
            )
        elif base_name == "MICN":
            model = MICN(
                n_steps=self.window_size, n_features=9, 
                n_pred_steps=self.pred_steps, n_pred_features=9,
                n_layers=config.get('n_layers', 2),
                d_model=config['d_model'],
                conv_kernel=config.get('conv_kernel', [7, 9]), 
                dropout=config.get('dropout', 0.1),
                epochs=1
            )
        else:
            raise ValueError(f"Forecaster no soportado: {base_name}")
            
        model.load(path)
        return model

    def _compute_metrics(self, pred_vals: np.ndarray, real_vals: np.ndarray) -> dict:
        mae = float(np.mean(np.abs(pred_vals - real_vals)))
        mse = float(np.mean(np.square(pred_vals - real_vals)))
        rmse = float(np.sqrt(mse))
        return {"MAE": mae, "MSE": mse, "RMSE": rmse}

    def _get_imputer_window(self) -> int:
        return int(DEFAULT_MODEL_KWARGS.get(self.imputer_name, {}).get('window_size', 0))

    def run_experiment(self):
        print(f"\n[1/4] Cargando conjunto de datos Test (Ventana Forecaster: {self.window_size})...")
        loader = DataLoader(self.data_path, window_size=self.window_size)
        _, _, test_set = loader.get_splits()
        
        print("\n[2/4] Evaluando Predicción sobre datos ORIGINALES (con NaNs)...")
        forecaster = self._load_forecaster()
        
        res_raw = forecaster.predict(test_set)
        pred_raw = np.array(res_raw["forecasting"]) if isinstance(res_raw, dict) else np.array(res_raw)
        
        # Adapt future observations to the prediction steps
        real_future = np.array(test_set["X_pred"])[:, :self.pred_steps, :]
        
        eval_mask = ~np.isnan(real_future)
        metrics_raw = self._compute_metrics(pred_raw[eval_mask], real_future[eval_mask])
        
        print(f"\n[3/4] Imputando datos de entrada con el modelo de limpieza ({self.imputer_name})...")
        imputer = self._load_imputer()
        imputer_w = self._get_imputer_window()
        
        X = np.array(test_set["X"]) 
        N_samples, forecaster_w, n_features = X.shape
        
        if forecaster_w % imputer_w != 0:
            raise ValueError(f"Dimensión incompatible: La ventana de predicción ({forecaster_w}) "
                             f"no es divisible por la ventana de imputación ({imputer_w}).")
            
        factor = forecaster_w // imputer_w
        print(f"      * Troceando ventanas de {forecaster_w} en {factor} bloques de {imputer_w} pasos...")
        
        # Minibatch processing for GPU memory management
        batch_size = 1024 
        imputed_chunks_list = []
        
        import torch
        
        for start_idx in range(0, N_samples, batch_size):
            end_idx = min(start_idx + batch_size, N_samples)

            # Chunking            
            X_batch= X[start_idx:end_idx]
            batch_samples = X_batch.shape[0]
            X_batch_reshaped = X_batch.reshape(batch_samples * factor, imputer_w, n_features)
            
            res_batch = imputer.impute({"X": X_batch_reshaped})
            X_batch_imputed = np.array(res_batch["imputation"]) if isinstance(res_batch, dict) else np.array(res_batch)
            
            if X_batch_imputed.ndim == 4:
                X_batch_imputed = X_batch_imputed.mean(axis=1)
            
            # Rebuild the original shape
            X_batch_final = X_batch_imputed.reshape(batch_samples, forecaster_w, n_features)
            imputed_chunks_list.append(X_batch_final)
            torch.cuda.empty_cache()
            print(f"        - Progreso Imputación: {end_idx}/{N_samples} muestras procesadas...")

        # Concatenate all imputed chunks to form the final cleaned dataset
        X_full = np.concatenate(imputed_chunks_list, axis=0)
            
        np.save(os.path.join(self.repo_root, "data", f"X_test_limpio_{self.imputer_name}_w{self.window_size}.npy"), X_full)
        print("      Datos ensamblados, limpios y guardados en memoria.")
        
        print("\n[4/4] Evaluando Predicción sobre datos LIMPIOS...")
        test_set_limpio = {
            "X": X_full,                     
            "X_pred": real_future
        }
        
        res_clean = forecaster.predict(test_set_limpio)
        pred_clean = np.array(res_clean["forecasting"]) if isinstance(res_clean, dict) else np.array(res_clean)
        metrics_clean = self._compute_metrics(pred_clean[eval_mask], real_future[eval_mask])
        
        print("\n" + "="*60)
        print(f"RESULTADOS DEL EXPERIMENTO PIPELINE")
        print(f"   Imputador:  {self.imputer_name} (Ventana propia: {imputer_w})")
        print(f"   Forecaster: {self.forecaster_name} (Ventana propia: {forecaster_w})")
        print("="*60)
        print(f"Forecasting sobre datos CRUDOS (NaNs):")
        print(f"   MAE:  {metrics_raw['MAE']:.5f}")
        print(f"   MSE:  {metrics_raw['MSE']:.5f}")
        print(f"   RMSE: {metrics_raw['RMSE']:.5f}")
        print("-" * 60)
        print(f"Forecasting sobre datos IMPUTADOS:")
        print(f"   MAE:  {metrics_clean['MAE']:.5f}")
        print(f"   MSE:  {metrics_clean['MSE']:.5f}")
        print(f"   RMSE: {metrics_clean['RMSE']:.5f}")
        
        diff = ((metrics_raw['MAE'] - metrics_clean['MAE']) / metrics_raw['MAE']) * 100
        print("-" * 60)
        if diff > 0:
            print(f"La limpieza de datos MEJORÓ el MAE en un {diff:.2f} %")
        else:
            print(f"El modelo demostró ALTA ROBUSTEZ. Variación: {diff:.2f} %")
        print("="*60)

        # Save to txt file for record keeping
        results_log_path = os.path.join(self.repo_root, "logs_experiments", "pipeline_metrics_summary.txt")
        os.makedirs(os.path.dirname(results_log_path), exist_ok=True)
        
        output_txt = (
            f"Pipeline: {self.imputer_name} -> {self.forecaster_name}:\n"
            f"MAE: {metrics_clean['MAE']:.5f}\n"
            f"MSE: {metrics_clean['MSE']:.5f}\n"
            f"RMSE: {metrics_clean['RMSE']:.5f}\n"
            f"{'='*40}\n\n"
        )
        
        with open(results_log_path, "a", encoding="utf-8") as f:
            f.write(output_txt)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pipeline de Evaluación: Imputación + Forecasting')
    parser.add_argument('--imputer', type=str, default='SAITS', help='Clave exacta del modelo en model_configs.py (ej. SAITS o CSDI)')
    parser.add_argument('--forecaster', type=str, required=True, help='Clave exacta del modelo en model_configs.py (ej. DLinear_w384_ps192)')
    parser.add_argument('--pred_steps', type=int, default=192, help='Pasos a predecir')
    
    args = parser.parse_args()
    
    pipeline = PipelineEvaluator(
        imputer_name=args.imputer,
        forecaster_name=args.forecaster,
        pred_steps=args.pred_steps
    )
    pipeline.run_experiment()