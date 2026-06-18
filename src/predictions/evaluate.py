# src/predictions/evaluate.py
# usage: python src/predictions/evaluate.py --model DLinear --mode nn --window_size 192

import os
import ast
import numpy as np
from typing import Any
from dotenv import load_dotenv
load_dotenv() 

from predictions.data_loader import DataLoader
from pypots.imputation import SAITS, CSDI, LOCF, Mean
from pypots.forecasting.micn import MICN
from pypots.forecasting.transformer import Transformer
from pypots.forecasting.dlinear import DLinear

class Evaluator:
    def __init__(self, target_model: str, mode: str, window: int):
        self.target_model = target_model 
        self.mode = mode
        
        self.repo_root = os.environ.get("PROJECT_ROOT", os.getcwd())
        self.model_logs = os.environ.get("BEST_MODEL_LOG", "best_models_summary.txt")
        self.test_logs = os.environ.get("TEST_RESULTS_LOG", "test_metrics_summary.txt")
        self.results_file = os.path.join(self.repo_root, "logs_experiments", self.test_logs)
        self.summary_file = os.path.join(self.repo_root, "logs_experiments", self.model_logs)
        self.data_path = os.path.join(self.repo_root, "data", os.environ.get("PARQUET_MODEL", "df_pypots.parquet"))
        
        self.task_type = os.environ.get("TASK_TYPE", "imputation").lower()
        self.pred_steps = int(os.environ.get("PRED_STEPS", 96))
        
        self.best_params = {}
        self.model_path = ""
        self.window_size = window
        
        if self.target_model and self.mode in ('nn', 'all'):
            self._parse_summary()
            
        if not self.window_size:
            raise ValueError("No se pudo determinar el window_size. Pasa el argumento --window_size en la terminal.")
            
        print(f"Cargando datos para evaluación (Ventana: {self.window_size})...")
        loader = DataLoader(self.data_path, window_size=self.window_size)
        _, _, self.test_set = loader.get_splits()

    def _parse_summary(self):
        """ Reads best_models_summary.txt to find the best configuration and model path for the target model."""
        if not os.path.exists(self.summary_file):
            raise FileNotFoundError(f"No se encuentra el archivo de resumen en: {self.summary_file}")
            
        with open(self.summary_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{self.target_model}:"):
                found = True
                for j in range(1, 6):
                    if i+j >= len(lines): break
                    if "Best params:" in lines[i+j]:
                        param_str = lines[i+j].split("Best params: ")[1].strip()
                        self.best_params = ast.literal_eval(param_str)
                    if "Path:" in lines[i+j]:
                        base_dir = lines[i+j].split("Path: ")[1].strip()
                        if not base_dir.endswith(".pypots"):
                            base_dir = os.path.join(base_dir, f"{self.target_model}.pypots")
                        self.model_path = base_dir
                break
                
        if not found:
            raise ValueError(f"No se encontró el modelo {self.target_model} en el resumen.")
            
        self.window_size = self.best_params.get('window_size', self.window_size) # .get in case window_size is not explicitly saved in the summary
        print(f"[{self.target_model}] Configuración óptima recuperada automáticamente.")
        
    def _load_model(self) -> Any:
        """Builds the model architecture based on the best parameters and loads the trained weights from the .pypots file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encuentra el archivo .pypots en: {self.model_path}")
            
        print(f"Reconstruyendo arquitectura {self.target_model} e inyectando pesos entrenados...")
        
        if self.target_model == "SAITS":
            model = SAITS(
                n_steps=self.window_size, n_features=9, n_layers=self.best_params['n_layers'],
                d_model=self.best_params['d_model'], d_ffn=self.best_params.get('d_ffn', self.best_params['d_model']),
                n_heads=self.best_params['n_heads'], d_k=self.best_params['d_model'] // self.best_params['n_heads'],
                d_v=self.best_params['d_model'] // self.best_params['n_heads'], dropout=self.best_params['dropout'],
                epochs=1, batch_size=self.best_params.get('batch_size', 64)
            )
        elif self.target_model == "CSDI":
            model = CSDI(
                n_steps=self.window_size, n_features=9, n_layers=self.best_params['n_layers'],
                n_heads=self.best_params['n_heads'], n_channels=self.best_params['n_channels'],
                d_time_embedding=self.best_params.get('d_time_embedding', 128),
                d_feature_embedding=self.best_params.get('d_feature_embedding', 128),
                d_diffusion_embedding=self.best_params.get('d_diffusion_embedding', 128),
                n_diffusion_steps=self.best_params.get('n_diffusion_steps', 30),
                target_strategy=self.best_params.get('target_strategy', 'random'),
                epochs=1, batch_size=self.best_params.get('batch_size', 64)
            )
        elif self.target_model == "MICN":
            model = MICN(
                n_steps=self.window_size, 
                n_features=9, 
                n_pred_steps=self.pred_steps, 
                n_pred_features=9,
                n_layers=self.best_params.get('n_layers', 2),
                d_model=self.best_params['d_model'],
                conv_kernel=self.best_params.get('conv_kernel', [7, 9]), 
                dropout=self.best_params.get('dropout', 0.1),
                epochs=1, 
                batch_size=self.best_params.get('batch_size', 64)
            )
        elif self.target_model == "Transformer":
            model = Transformer(
                n_steps=self.window_size, n_features=9, 
                n_pred_steps=self.pred_steps, n_pred_features=9,
                n_encoder_layers=self.best_params.get('n_layers', 2),
                n_decoder_layers=self.best_params.get('n_layers', 2), 
                d_model=self.best_params['d_model'],
                d_ffn=self.best_params.get('d_ffn', 256), 
                n_heads=self.best_params['n_heads'],
                d_k=self.best_params['d_model'] // self.best_params['n_heads'],
                d_v=self.best_params['d_model'] // self.best_params['n_heads'],
                dropout=self.best_params['dropout'],
                epochs=1, batch_size=self.best_params.get('batch_size', 64)
            )
        elif self.target_model == "DLinear":
            model = DLinear(
                n_steps=self.window_size, n_features=9, 
                n_pred_steps=self.pred_steps, n_pred_features=9,
                moving_avg_window_size=self.best_params.get('moving_avg_window_size', 25),
                d_model=self.best_params.get('d_model', 128),  
                epochs=1, batch_size=self.best_params.get('batch_size', 64)
            )
        else:
            raise ValueError(f"Modelo no soportado: {self.target_model}")
            
        model.load(self.model_path)
        return model

    def _compute_metrics(self, pred_vals: np.ndarray, real_vals: np.ndarray) -> dict:
        """Computes MAE, MSE, and RMSE between the predicted values and the real values."""
        mae = float(np.mean(np.abs(pred_vals - real_vals)))
        mse = float(np.mean(np.square(pred_vals - real_vals)))
        rmse = float(np.sqrt(mse))
        return {"MAE": mae, "MSE": mse, "RMSE": rmse}

    def _log_and_save_results(self, model_name: str, metrics: dict):
        """Logs the evaluation metrics to the console and appends them to a summary file."""
        output = (
            f"{model_name}:\n"
            f"MAE: {metrics['MAE']:.5f}\n"
            f"MSE: {metrics['MSE']:.5f}\n"
            f"RMSE: {metrics['RMSE']:.5f}\n"
            f"{'='*40}\n\n"
        )
        print(output)
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(output)

    def evaluate_model(self):
        """Launches the inference process on the Test set and computes the evaluation metrics."""
        model = self._load_model()
        print("Realizando inferencia sobre el conjunto de Test...")
        test_set = self.test_set
        assert test_set is not None
        
        if self.task_type == 'imputation':
            results = model.impute(test_set)
            imputed_data = np.array(results["imputation"]) if isinstance(results, dict) else np.array(results)
            if imputed_data.ndim == 4: imputed_data = imputed_data.mean(axis=1)
                
            real_data = np.array(test_set["X_ori"])
            input_data = np.array(test_set["X"])
            eval_mask = np.isnan(input_data) & ~np.isnan(real_data)
            pred_vals = imputed_data[eval_mask]
            real_vals = real_data[eval_mask]
            
        elif self.task_type == 'forecasting':
            results = model.predict(test_set)
            predicted_data = np.array(results["forecasting"]) if isinstance(results, dict) else np.array(results)
            real_data = np.array(test_set["X_pred"])
            
            eval_mask = ~np.isnan(real_data)
            pred_vals = predicted_data[eval_mask]
            real_vals = real_data[eval_mask]
        else:
            raise ValueError(f"Tipo de tarea no soportado: {self.task_type}")

        metrics = self._compute_metrics(pred_vals, real_vals)
        self._log_and_save_results(f"Red Neuronal: {self.target_model}", metrics)

    def evaluate_mean(self):
        if self.task_type == 'forecasting': return
        print("Realizando inferencia sobre el conjunto de Test con la Media Global...")
        mean_model = Mean()
        test_set = self.test_set
        assert test_set is not None
        results = mean_model.impute(test_set)
        mean_imputed = np.array(results["imputation"]) if isinstance(results, dict) else np.array(results)
        
        real_data = np.array(test_set["X_ori"])
        input_data = np.array(test_set["X"])
        eval_mask = np.isnan(input_data) & ~np.isnan(real_data)
        
        metrics = self._compute_metrics(mean_imputed[eval_mask], real_data[eval_mask])
        self._log_and_save_results("Baseline: MEDIA GLOBAL", metrics)

    def evaluate_locf(self):
        if self.task_type == 'forecasting': return
        print("Realizando inferencia sobre el conjunto de Test con LOCF...")
        locf_model = LOCF()
        test_set = self.test_set
        assert test_set is not None
        results = locf_model.impute(test_set)
        locf_imputed = np.array(results["imputation"]) if isinstance(results, dict) else np.array(results)
        
        real_data = np.array(test_set["X_ori"])
        input_data = np.array(test_set["X"])
        eval_mask = np.isnan(input_data) & ~np.isnan(real_data)
        
        metrics = self._compute_metrics(locf_imputed[eval_mask], real_data[eval_mask])
        self._log_and_save_results("Baseline: LOCF", metrics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluación de Modelos en Test')
    parser.add_argument('--model', type=str, choices=['SAITS', 'CSDI', 'MICN', 'Transformer', 'DLinear'])
    parser.add_argument('--window_size', type=int, help='Ventana temporal')
    parser.add_argument('--mode', type=str, default='nn', choices=['nn', 'baselines', 'all'], help='Modo de evaluación')
    args = parser.parse_args()
    
    if args.mode in ('nn', 'all') and not args.model:
        raise ValueError('Debes especificar --model cuando mode es nn o all')
        
    evaluator = Evaluator(args.model, args.mode, args.window_size)
    
    if args.mode in ('nn', 'all'):
        evaluator.evaluate_model()
        
    if args.mode in ('baselines', 'all'):
        evaluator.evaluate_mean()
        evaluator.evaluate_locf()