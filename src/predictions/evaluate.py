# predictions/evaluate.py
# usage: python src/predictions/evaluate.py --model_path /path/to/best_model.pkl --window_size <window_size_used_IN THE BEST MODEL>
import os
import numpy as np
import torch
from predictions.data_loader import DataLoader

class Evaluator:
    """
    Evaluates trained models on the Test Set
    Calculates MAE, MSE, RMSE for imputation tasks using the original data as ground truth (X_ori)
    TODO: Will be adapted to evaluate forecasting models
    """
    def __init__(self, model_path: str, data_path: str, window_size: int):
        self.model_path = model_path
        self.data_path = data_path
        self.window_size = window_size 
        
        print("Cargando datos para evaluación...")
        loader = DataLoader(self.data_path, window_size=self.window_size)
        _, _, self.test_set = loader.get_splits()
        
    def load_model(self):
        """Load the trained model (.pypots) from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encuentra el modelo en: {self.model_path}")
            
        print(f"Cargando modelo PyPOTS desde: {self.model_path}")
        model = torch.load(self.model_path)
        return model

    def evaluate_imputation(self):
        """
        Calculates MAE, MSE, RMSE for imputation tasks
        TODO: Adapt for forecasting models
        """

        model = self.load_model()
        print("Realizando inferencia sobre el Test Set...")
        results = model.impute(self.test_set)
        
        # Check output pypot format and extract imputed data
        if isinstance(results, dict):
            imputed_data = np.array(results["imputation"])
        else:
            imputed_data = np.array(results)
        
        real_data = np.array(self.test_set["X_ori"])  # original data
        input_data = np.array(self.test_set["X"])     # data with artificially introduced NaNs
        eval_mask = np.isnan(input_data) & ~np.isnan(real_data) # evaluate only in artificially introduced NaN positions where there is ground truth
        
        print("\n" + "="*40)
        print(" RESULTADOS DE EVALUACIÓN (IMPUTACIÓN)")
        print("="*40)
        
        pred_vals = imputed_data[eval_mask]
        real_vals = real_data[eval_mask]
        
        # Calculate metrics
        mae = float(np.mean(np.abs(pred_vals - real_vals)))
        mse = float(np.mean(np.square(pred_vals - real_vals)))
        rmse = float(np.sqrt(mse))
        
        print(f"MAE  (Error Absoluto Medio): {mae:.5f}")
        print(f"MSE  (Error Cuadrático Medio): {mse:.5f}")
        print(f"RMSE (Raíz del Error Cuadrático): {rmse:.5f}")
        print("="*40 + "\n")
        return {"MAE": mae, "MSE": mse, "RMSE": rmse}


#################### Execution flow
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluación de Modelos en Test')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta absoluta al modelo entrenado (.pypots)')
    parser.add_argument('--window_size', type=int, required=True, help='Ventana con la que se entrenó el modelo')
    args = parser.parse_args()
    
    # Data path
    data_folder = os.environ.get("DATA_FOLDER_PATH_IPYNB", "../../data/")
    df_original = os.environ.get("PARQUET_MODEL", "df_pypots.parquet")
    data_path = os.path.join(data_folder, df_original)
    
    # Launch evaluation
    evaluator = Evaluator(args.model_path, data_path, args.window_size)
    evaluator.evaluate_imputation()