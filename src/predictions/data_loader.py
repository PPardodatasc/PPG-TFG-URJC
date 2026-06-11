import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
load_dotenv()

DATA_AMOUNT = float(os.environ.get("DATA_AMOUNT", 0.4)) 
MISSING_RATE = float(os.environ.get("MISSING_RATE", 0.15)) 
TASK_TYPE = os.environ.get("TASK_TYPE", "imputation").lower()
PRED_STEPS = int(os.environ.get("PRED_STEPS", 96)) # 96 = 24h, 192 = 48h, 672 = 1 week

class DataLoader:
    """
    Loads and processes parking sensor data from a Parquet file.
    Automatically adapts its sliding window logic for Imputation or Forecasting.
    """
    def __init__(self, filepath, window_size):
        self.filepath = filepath
        self.window_size = window_size
        self.features = [
            'VehiclePresent', 
            'lon', 'lat',     
            'Hour_sin', 'Hour_cos', 
            'WeekDay_sin', 'WeekDay_cos', 
            'Month_sin', 'Month_cos'
        ]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.process_data() 


    def _create_sliding_windows_imputation(self, df):
        """ 
        Sliding windows for Imputation
        """
        X_list = []
        for device_id, group in df.groupby('DeviceId'):
            data = group[self.features].values 
            for i in range(len(data) - self.window_size + 1):
                X_list.append(data[i : i + self.window_size])
        return np.array(X_list)


    def _create_sliding_windows_forecasting(self, df):
        """
        Sliding windows for Forecasting (X = past, X_pred = future)
        """
        X_list = []
        Y_list = []
        for device_id, group in df.groupby('DeviceId'):
            data = group[self.features].values 
            # Ensure enough data points for both input and prediction windows
            for i in range(len(data) - self.window_size - PRED_STEPS + 1):
                X_list.append(data[i : i + self.window_size])
                Y_list.append(data[i + self.window_size : i + self.window_size + PRED_STEPS])
                
        return np.array(X_list), np.array(Y_list)


    def _mask_data(self, X, missing_rate=0.15):
        """
        Artificial masking for evaluation in validation and test (only for imputation)
        """
        X_masked = np.copy(X)
        observed_indices = np.where(~np.isnan(X))
        n_observed = len(observed_indices[0])
        n_hide = int(n_observed * missing_rate)
        hide_idx = np.random.choice(n_observed, n_hide, replace=False) 
        
        X_masked[observed_indices[0][hide_idx], 
                    observed_indices[1][hide_idx], 
                    observed_indices[2][hide_idx]] = np.nan
        return X_masked
    

    def process_data(self):
        """
        Orchestrates the data loading based on TASK_TYPE
        """
        print(f"[{self.__class__.__name__}] Modo: {TASK_TYPE.upper()} | Cargando datos...")
        df = pd.read_parquet(self.filepath)

        np.random.seed(33) 
        unique_devices = df['DeviceId'].unique()
        selected_devices = np.random.choice(unique_devices, size=int(len(unique_devices) * DATA_AMOUNT), replace=False) 
        df = df[df['DeviceId'].isin(selected_devices)].reset_index(drop=True)
        print(f"[{self.__class__.__name__}] Sensores usados: {len(selected_devices)}/{len(unique_devices)} (Seed 33 preservada)")
        
        df['VehiclePresent'] = df['VehiclePresent'].astype(float)
        df.fillna(value=np.nan, inplace=True) 

        # Data splits
        ts_unique = np.sort(df['Timestamp'].unique())
        n_ts = len(ts_unique)
        train_cut = ts_unique[int(n_ts * 0.60)]
        val_cut = ts_unique[int(n_ts * 0.80)]
        
        df_train = df[df['Timestamp'] <= train_cut]
        df_val = df[(df['Timestamp'] > train_cut) & (df['Timestamp'] <= val_cut)]
        df_test = df[df['Timestamp'] > val_cut]
        
        print(f"[{self.__class__.__name__}] Generando tensores (Ventana Entrada: {self.window_size}" + 
              (f", Ventana Predicción: {PRED_STEPS})" if TASK_TYPE == 'forecasting' else ")..."))

        # Bifurcation acording to TASK_TYPE
        if TASK_TYPE == 'imputation':
            X_train = self._create_sliding_windows_imputation(df_train)
            X_val = self._create_sliding_windows_imputation(df_val)
            X_test = self._create_sliding_windows_imputation(df_test)
            
            X_val_masked = self._mask_data(X_val, MISSING_RATE)
            X_test_masked = self._mask_data(X_test, MISSING_RATE)

            self.train_dataset = {"X": X_train}
            self.val_dataset = {"X": X_val_masked, "X_ori": X_val}
            self.test_dataset = {"X": X_test_masked, "X_ori": X_test}
            
        elif TASK_TYPE == 'forecasting':
            X_train, Y_train = self._create_sliding_windows_forecasting(df_train)
            X_val, Y_val = self._create_sliding_windows_forecasting(df_val)
            X_test, Y_test = self._create_sliding_windows_forecasting(df_test)
            
            # PyPOTS forecasting API expects X for inputs and X_pred for the future horizon.
            self.train_dataset = {"X": X_train, "X_pred": Y_train}
            self.val_dataset = {"X": X_val, "X_pred": Y_val}
            self.test_dataset = {"X": X_test, "X_pred": Y_test}
        else:
            raise ValueError(f"TASK_TYPE no soportado: {TASK_TYPE}")

        assert self.train_dataset is not None
        assert self.val_dataset is not None
        assert self.test_dataset is not None
            
        print(f"[{self.__class__.__name__}] Particiones - Train X: {self.train_dataset['X'].shape}, Val X: {self.val_dataset['X'].shape}, Test X: {self.test_dataset['X'].shape}")

    def get_splits(self):
        return self.train_dataset, self.val_dataset, self.test_dataset