# predictions/train.py
import os
import numpy as np
from typing import cast
from dotenv import load_dotenv
load_dotenv()

from predictions.data_loader import DataLoader
from predictions.model_configs import DEFAULT_MODEL_KWARGS
from pypots.optim import Adam
from pypots.imputation import SAITS, CSDI
from pypots.forecasting.dlinear import DLinear
from pypots.forecasting.micn import MICN
from pypots.forecasting.transformer import Transformer

class Trainer:
    """
    Comprises the training logic for both imputation and forecasting models.
    """
    def __init__(self, model_name: str, repo_root: str, data_path: str):
        self.model_name = model_name
        self.repo_root = repo_root
        self.data_path = data_path
        self.task_type = os.environ.get("TASK_TYPE", "imputation").lower()
        self.pred_steps = int(os.environ.get("PRED_STEPS", 96))

    def _build_model(self, config: dict, n_features: int, save_dir: str):
        """
        Builds the PyPOTS model configurations. 
        The final_config is the result of merging the default config (model_configs.py) with the hyperparameter search config (configs/).
        """
        
        config['optimizer'] = Adam(lr=config['lr'])
        config['saving_path'] = save_dir

        if self.model_name == 'SAITS':
            saits_kwargs = {
                'n_steps': config['window_size'],
                'n_features': n_features,
                'n_layers': config['n_layers'],
                'd_model': config['d_model'],
                'd_ffn': config.get('d_ffn', config['d_model']),
                'n_heads': config['n_heads'],
                'd_k': config['d_model'] // config['n_heads'],
                'd_v': config['d_model'] // config['n_heads'],
                'dropout': config['dropout'],
                'patience': config['patience'],
                'model_saving_strategy': config['model_saving_strategy'],
                'batch_size': config['batch_size'],
                'epochs': config['epochs'],
                'optimizer': config['optimizer'],
                'saving_path': config['saving_path']
            }
            return SAITS(**saits_kwargs)

        if self.model_name == 'CSDI':
            csdi_kwargs = {
                'n_steps': config['window_size'],
                'n_features': n_features,
                'n_layers': config['n_layers'],
                'n_channels': config['n_channels'],
                'n_heads': config['n_heads'],
                'target_strategy': config['target_strategy'],
                'n_diffusion_steps': config['n_diffusion_steps'],
                'patience': config['patience'],
                'model_saving_strategy': config['model_saving_strategy'],
                'd_time_embedding': config['d_time_embedding'],
                'd_feature_embedding': config['d_feature_embedding'],
                'd_diffusion_embedding': config['d_diffusion_embedding'],
                'batch_size': config['batch_size'],
                'epochs': config['epochs'],
                'optimizer': config['optimizer'],
                'saving_path': config['saving_path']
            }
            return CSDI(**csdi_kwargs)

        if self.model_name == 'MICN':
            kwargs = {
                'n_steps': config.get('window_size', 192),
                'n_features': n_features,
                'n_pred_steps': self.pred_steps,
                'n_pred_features': n_features,
                'n_layers': config.get('n_layers', 2),
                'd_model': config['d_model'],
                'conv_kernel': config.get('conv_kernel', [7, 9]), 
                'dropout': config.get('dropout', 0.1),
                'epochs': config.get('epochs', 20),
                'batch_size': config.get('batch_size', 64),
                'optimizer': config['optimizer'],
                'saving_path': config['saving_path'],
                'patience': config['patience'],
                'model_saving_strategy': config.get('model_saving_strategy', 'best')
            }
            return MICN(**kwargs)

        if self.model_name == 'Transformer':
            transformer_kwargs = {
                'n_steps': config['window_size'],
                'n_features': n_features,
                'n_pred_steps': self.pred_steps,
                'n_pred_features': n_features,
                'n_encoder_layers': config['n_layers'], 
                'n_decoder_layers': config['n_layers'], 
                'd_model': config['d_model'],
                'd_ffn': config.get('d_ffn', config['d_model'] * 2),
                'n_heads': config['n_heads'],
                'd_k': config['d_model'] // config['n_heads'],
                'd_v': config['d_model'] // config['n_heads'],
                'dropout': config['dropout'],
                'patience': config['patience'],
                'model_saving_strategy': config.get('model_saving_strategy', 'best'),
                'batch_size': config['batch_size'],
                'epochs': config['epochs'],
                'optimizer': config['optimizer'],
                'saving_path': config['saving_path']
            }
            return Transformer(**transformer_kwargs)

        if self.model_name == 'DLinear':
            dlinear_kwargs = {
                'n_steps': config['window_size'],
                'n_features': n_features,
                'n_pred_steps': self.pred_steps,
                'n_pred_features': n_features,
                'moving_avg_window_size': config.get('moving_avg_window_size', 25),
                'd_model': config.get('d_model', 128),
                'patience': config['patience'],
                'model_saving_strategy': config.get('model_saving_strategy', 'best'),
                'batch_size': config['batch_size'],
                'epochs': config['epochs'],
                'optimizer': config['optimizer'],
                'saving_path': config['saving_path']
            }
            return DLinear(**dlinear_kwargs)

        raise ValueError(f"Modelo no soportado: {self.model_name}")

    def _get_path(self, config: dict, w_size: int) -> str:
        """
        Constructs the saving path for the model based on the configuration.
        """
        lr_str = f"{config['lr']:.5f}"
        base_name = f"{self.model_name}_w{w_size}_e{config['epochs']}_b{config['batch_size']}_lr{lr_str}"
        
        if self.model_name == 'SAITS':
            spec_name = f"_lyr{config['n_layers']}_dm{config['d_model']}_hd{config['n_heads']}_dp{config['dropout']}"
        elif self.model_name == 'CSDI':
            spec_name = (
                f"_lyr{config['n_layers']}_ch{config['n_channels']}_hd{config['n_heads']}"
                f"_dp{config['dropout']}_ds{config['n_diffusion_steps']}"
                f"_te{config['d_time_embedding']}_fe{config['d_feature_embedding']}"
            )
        elif self.model_name == 'MICN':
            spec_name = f"_ps{self.pred_steps}_lyr{config.get('n_layers',2)}_dm{config['d_model']}"
        elif self.model_name == 'Transformer':
            spec_name = f"_ps{self.pred_steps}_lyr{config['n_layers']}_dm{config['d_model']}_hd{config['n_heads']}"
        elif self.model_name == 'DLinear':
            spec_name = f"_ps{self.pred_steps}_ma{config.get('moving_avg_window_size',25)}"
        else:
            spec_name = ""
        
        # Build path
        exp_name = base_name + spec_name
        task_folder = "Imputation" if self.task_type == 'imputation' else "Forecasting"
        save_dir = os.path.join(self.repo_root, "logs_experiments", task_folder, self.model_name, exp_name)
        return save_dir

    def train(self, config: dict) -> tuple:
        """
        Trains the model with the given configuration and evaluates it on the validation set.
        Returns the path where the model is saved and the MAE obtained in validation.
        """
        final_config = DEFAULT_MODEL_KWARGS.get(self.model_name, {}).copy()
        final_config.update(config)
        
        w_size = final_config['window_size']

        # Get data splits and number of features for model building
        loader = DataLoader(self.data_path, window_size=w_size)
        train_set, val_set, _ = loader.get_splits()
        n_features = len(loader.features)

        save_dir = self._get_path(final_config, w_size) # Save path construction
        model = self._build_model(final_config, n_features, save_dir) # Build the model with the final configuration

        # Training and validation
        model.fit(train_set=cast(dict, train_set), val_set=cast(dict, val_set))
        
        if self.task_type == 'imputation':
            imputation_results = model.impute(val_set)
            if isinstance(imputation_results, dict):
                predicted_data = np.array(imputation_results["imputation"])
            else:
                predicted_data = np.array(imputation_results)
            
            if predicted_data.ndim == 4: 
                predicted_data = predicted_data.mean(axis=1) 
            
            real_data = np.array(val_set["X_ori"])  
            input_data = np.array(val_set["X"])     
            
            eval_mask = np.isnan(input_data) & ~np.isnan(real_data) 
            pred_vals = predicted_data[eval_mask]
            real_vals = real_data[eval_mask]
            
        elif self.task_type == 'forecasting':
            results = model.predict(val_set)
            predicted_data = np.array(results["forecasting"]) if isinstance(results, dict) else np.array(results)
            
            real_vals = np.array(val_set["X_pred"])
            eval_mask = ~np.isnan(real_vals)
            pred_vals = predicted_data[eval_mask]
            real_vals = real_vals[eval_mask]
        
        mae = float(np.mean(np.abs(pred_vals - real_vals)))
        return save_dir, mae