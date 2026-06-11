# model_configs.py
DEFAULT_MODEL_KWARGS = {
    # IMPUTATION MODELS
    # Best params: {'window_size': 48, 'batch_size': 64, 'lr': 0.00047295728151429, 'n_layers': 2, 'd_model': 128, 'n_heads': 4, 'dropout': 0.2}
    'SAITS': {
        'epochs': 20,           
        'window_size': 48,       
        'batch_size': 64,        
        'lr': 0.00047295728151429,             
        'n_layers': 2,
        'd_model': 128,
        'n_heads': 4,
        'dropout': 0.2,
        'patience': 2,
        'model_saving_strategy': 'best'
    },
    # Best params: {'window_size': 96, 'batch_size': 128, 'lr': 0.001289884852043542, 'n_layers': 3, 'n_channels': 128, 'n_heads': 2, 'dropout': 0.2}
    'CSDI': {
        'epochs': 20,
        'window_size': 96,
        'batch_size': 128,
        'lr': 0.001289884852043542,
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
        'dropout': 0.2
    },

    #####  FORECASTING MODELS

    # FORECASTING 24H, WINDOW SIZE 48H
    # Best params: {'window_size':192, 'batch_size': 128, 'lr': 0.00376, 'n_layers': 2, 'd_model': 64, 'conv_kernel': [7, 9], 'dropout': 0.1}
    'MICN_w192_ps96': {
        'epochs': 20,
        'window_size': 192,      
        'd_model': 64,           
        'n_layers': 2,           
        'conv_kernel': [7, 9],   
        'dropout': 0.1,          
        'batch_size': 128,       
        'lr': 0.00376,           
        'patience': 2,
        'model_saving_strategy': 'best'
    },
    # Best params: {'window_size': 192, 'batch_size': 32, 'lr': 0.00011, 'n_layers': 2, 'd_model': 256, 'n_heads': 8, 'dropout': 0.1}
    'Transformer_w192_ps96': {
        'epochs': 20,
        'window_size': 192,      
        'd_model': 256,          
        'd_ffn': 256,           
        'n_heads': 8,            
        'n_layers': 2,           
        'n_encoder_layers': 2,
        'n_decoder_layers': 2,
        'dropout': 0.1,          
        'batch_size': 32,        
        'lr': 0.00011,           
        'patience': 2,
        'model_saving_strategy': 'best'
    },
    # Best params: {'window_size': 192, 'batch_size': 128, 'lr': 0.00663, 'n_layers': 2, 'd_model': 128, 'conv_kernel': [7, 9], 'dropout': 0.1}
    'DLinear_w192_ps96': {
        'epochs': 30,
        'window_size': 192,  
        'moving_avg_window_size': 99,  
        'd_model': 128,                
        'batch_size': 128,            
        'lr': 0.00663,                 
        'patience': 2,
        'model_saving_strategy': 'best'
    }

    # FORECASTING 48H, WINDOW SIZE 96H
}