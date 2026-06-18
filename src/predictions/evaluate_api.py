# src/predictions/evaluate_api.py
# usage: python evaluate_api.py --model SAITS --window_size 48 --mode nn
import os
from dotenv import load_dotenv
load_dotenv() 
from predictions.evaluate import Evaluator
from predictions.data_loader import DataLoader

class EvaluatorAPI(Evaluator):
    def __init__(self, target_model: str, mode: str, window: int):
        self.target_model = target_model
        self.base_model = target_model.split('_')[0]
        self.mode = mode
        self.window_size = window
        self.repo_root = os.environ.get("PROJECT_ROOT", os.getcwd())
        self.model_logs = os.environ.get("BEST_MODEL_LOG", "best_models_summary.txt")
        self.summary_file = os.path.join(self.repo_root, "logs_experiments", self.model_logs)
        self.test_logs = os.environ.get("OOT_RESULTS_LOG", "test_api_metrics_summary.txt")
        self.results_file = os.path.join(self.repo_root, "logs_experiments", self.test_logs)
        self.data_path = os.path.join(self.repo_root, "data", os.environ.get("PARQUET_API_MODEL", "df_pypots_api_oot.parquet"))
        self.task_type = os.environ.get("TASK_TYPE", "imputation").lower()
        self.pred_steps = int(os.environ.get("PRED_STEPS", 96))
        self.best_params = {}
        self.model_path = ""
        
        if self.target_model and self.mode in ('nn', 'all'):
            self._parse_summary()
            
        saved_window = self.best_params.get('window_size') or self.best_params.get('n_steps')
        if saved_window:
            self.window_size = saved_window

        if not self.window_size:
            raise ValueError("No se pudo determinar la ventana temporal.")
            
        print(f"[{self.target_model}] Cargando 100% de los datos OOT...")
        loader = DataLoader(self.data_path, window_size=self.window_size, is_oot=True) # oot flag true to avoid splitting the dataset into train/val/test
        _, _, self.test_set = loader.get_splits()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluación de Modelos en Test API (OOT)')
    parser.add_argument('--model', type=str, help='Nombre del modelo exacto como sale en el txt')
    parser.add_argument('--window_size', type=int, help='Ventana temporal (opcional si existe en txt)')
    parser.add_argument('--mode', type=str, default='nn', choices=['nn', 'baselines', 'all'])
    args = parser.parse_args()
    
    if args.mode in ('nn', 'all') and not args.model:
        raise ValueError('Debes especificar --model cuando mode es nn o all')
        
    evaluator = EvaluatorAPI(args.model, args.mode, args.window_size)
    
    if args.mode in ('nn', 'all'):
        evaluator.evaluate_model()
        
    if args.mode in ('baselines', 'all'):
        evaluator.evaluate_mean()
        evaluator.evaluate_locf()
