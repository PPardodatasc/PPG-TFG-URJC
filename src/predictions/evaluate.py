# src/predictions/evaluate.py
# usage: python src/predictions/evaluate.py --model SAITS --mode nn
import os
import ast
import numpy as np
from dotenv import load_dotenv
load_dotenv() 

from predictions.data_loader import DataLoader
from pypots.imputation import SAITS, CSDI, LOCF, Mean

class Evaluator:
    def __init__(self, target_model: str, mode: str, fallback_window: int = None):
        self.target_model = target_model 
        self.mode = mode
        
        self.repo_root = os.environ.get("PROJECT_ROOT", os.getcwd())
        self.model_logs = os.environ.get("BEST_MODEL_LOG", "best_models_summary.txt")
        self.test_logs = os.environ.get("TEST_RESULTS_LOG", "test_metrics_summary.txt")
        self.results_file = os.path.join(self.repo_root, "logs_experiments", self.test_logs)
        self.summary_file = os.path.join(self.repo_root, "logs_experiments", self.model_logs)
        self.data_path = os.path.join(self.repo_root, "data", os.environ.get("PARQUET_MODEL", "df_pypots.parquet"))
        
        self.best_params = {}
        self.model_path = ""
        self.window_size = fallback_window
        
        # Obtain info from summary .txt file
        if self.target_model and self.mode in ('nn', 'all'):
            self._parse_summary()
            
        if not self.window_size:
            raise ValueError("No se pudo determinar el window_size. Pasa el argumento --window_size en la terminal.")
            
        print(f"Cargando datos para evaluación (Ventana: {self.window_size})...")
        loader = DataLoader(self.data_path, window_size=self.window_size)
        _, _, self.test_set = loader.get_splits()

    def _parse_summary(self):
        """ Reads best_models_summary.txt to find the best configuration and model path for the target model.
        """
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
            
        self.window_size = self.best_params['window_size']
        print(f"[{self.target_model}] Configuración óptima recuperada automáticamente.")
        
    def _load_model(self):
        """Builds the model architecture based on the best parameters and loads the trained weights from the .pypots file.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encuentra el archivo .pypots en: {self.model_path}")
            
        print(f"Reconstruyendo arquitectura {self.target_model} e inyectando pesos entrenados...")
        
        if self.target_model == "SAITS":
            model = SAITS(
                n_steps=self.window_size,
                n_features=9,
                n_layers=self.best_params['n_layers'],
                d_model=self.best_params['d_model'],
                d_ffn=self.best_params.get('d_ffn', self.best_params['d_model']), # Corregido
                n_heads=self.best_params['n_heads'],
                d_k=self.best_params['d_model'] // self.best_params['n_heads'],   # NUEVO
                d_v=self.best_params['d_model'] // self.best_params['n_heads'],   # NUEVO
                dropout=self.best_params['dropout'],
                epochs=1, # Irrelevante para test
                batch_size=self.best_params.get('batch_size', 64)
            )

        elif self.target_model == "CSDI":
            model = CSDI(
                n_steps=self.window_size,
                n_features=9,
                n_layers=self.best_params['n_layers'],
                n_heads=self.best_params['n_heads'],
                n_channels=self.best_params['n_channels'],
                d_time_embedding=self.best_params.get('d_time_embedding', 128),
                d_feature_embedding=self.best_params.get('d_feature_embedding', 128),
                d_diffusion_embedding=self.best_params.get('d_diffusion_embedding', 128),
                n_diffusion_steps=self.best_params.get('n_diffusion_steps', 30),
                target_strategy=self.best_params.get('target_strategy', 'random'),
                epochs=1,
                batch_size=self.best_params.get('batch_size', 64)
            )
        else:
            raise ValueError(f"Modelo no soportado: {self.target_model}")
            
        model.load(self.model_path)
        return model

    def _get_eval_arrays(self):
        """
        Prepares the real values, input values, and evaluation mask for computing metrics. The evaluation mask identifies the originally missing entries in the test set.
        """
        real_data = np.array(self.test_set["X_ori"])
        input_data = np.array(self.test_set["X"])
        eval_mask = np.isnan(input_data) & ~np.isnan(real_data)
        return real_data, input_data, eval_mask

    def _compute_metrics(self, predicted_data: np.ndarray) -> dict:
        """
        Computes MAE, MSE, and RMSE between the imputed values and the real values in the test set, only for the originally missing entries.
        """
        real_data, _, eval_mask = self._get_eval_arrays()
        pred_vals = predicted_data[eval_mask]
        real_vals = real_data[eval_mask]

        mae = float(np.mean(np.abs(pred_vals - real_vals)))
        mse = float(np.mean(np.square(pred_vals - real_vals)))
        rmse = float(np.sqrt(mse))
        return {"MAE": mae, "MSE": mse, "RMSE": rmse}

    def _log_and_save_results(self, model_name: str, metrics: dict):
        """
        Logs the evaluation metrics to the console and appends them to a summary file for future reference.
        """
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

    def evaluate_imputation(self):
        """ 
        Launches the inference process on the Test set and computes the evaluation metrics
        """
        model = self._load_model()
        print("Realizando inferencia sobre el conjunto de Test...")
        results = model.impute(self.test_set)
        
        imputed_data = np.array(results["imputation"]) if isinstance(results, dict) else np.array(results)
        if imputed_data.ndim == 4:
            imputed_data = imputed_data.mean(axis=1)

        metrics = self._compute_metrics(imputed_data)
        self._log_and_save_results(f"Red Neuronal: {self.target_model}", metrics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluación de Modelos en Test')
    parser.add_argument('--model', type=str, choices=['SAITS', 'CSDI'], help='Nombre del modelo neuronal a evaluar')
    parser.add_argument('--window_size', type=int, help='Ventana temporal (requerido para evaluar baselines)') 
    parser.add_argument('--mode', type=str, default='nn', choices=['nn', 'baselines', 'all'], help='Modo de evaluación')
    args = parser.parse_args()
    
    if args.mode in ('nn', 'all') and not args.model:
        raise ValueError('Debes especificar --model (SAITS o CSDI) cuando mode es nn o all')
        
    evaluator = Evaluator(args.model, args.mode, args.window_size)
    
    if args.mode in ('nn', 'all'):
        evaluator.evaluate_imputation()