# model_configs.py
# TODO: This will be adapted with the best configurations found after hyperparameter tuning. For now, it contains the default configurations for all models.
DEFAULT_MODEL_KWARGS = {
    # IMPUTATION MODELS
    'SAITS': {
        'epochs': 20,           
        'window_size': 96,       
        'batch_size': 32,        
        'lr': 0.001,             
        'n_layers': 2,
        'd_model': 128,
        'n_heads': 2,
        'dropout': 0.1,
        'patience': 2,
        'model_saving_strategy': 'best'
    },
    'CSDI': {
        'epochs': 20,
        'window_size': 96,
        'batch_size': 32,
        'lr': 0.001,
        'n_layers': 3,
        'n_channels': 64,
        'n_heads': 2,
        'target_strategy': 'random',
        'n_diffusion_steps': 30,
        'patience': 2,
        'model_saving_strategy': 'best',
        'd_time_embedding': 128,
        'd_feature_embedding': 128,
        'd_diffusion_embedding': 128,
    },

    # FORECASTING MODELS
    'PatchTST': {
        'n_heads': 4,
        'd_model': 128,
        'd_ffn': 256,
        'patch_len': 16,     
        'stride': 8,         
        'dropout': 0.1,
        'patience': 5,
        'model_saving_strategy': 'best'
    },
    'iTransformer': {
        'd_model': 128,
        'd_ffn': 256,
        'n_heads': 4,
        'dropout': 0.1,
        'patience': 5,
        'model_saving_strategy': 'best'
    },
    'DLinear': {
        'moving_avg_window_size': 25, 
        'patience': 5,
        'model_saving_strategy': 'best'
    }
}