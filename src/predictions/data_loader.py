# data_loader.py
import pandas as pd
import numpy as np

class DataLoader:
    """
    Loads and processes parking sensor data from a Parquet file
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
        self._process_data() # triggers data loading and processing

    def _create_sliding_windows(self, df):
        """
        Transforms the df into 3D tensors using sliding windows per device.
        """
        X_list = []
        for device_id, group in df.groupby('DeviceId'):
            data = group[self.features].values 
            for i in range(len(data) - self.window_size + 1):
                X_list.append(data[i : i + self.window_size])
        return np.array(X_list)

    def _process_data(self):
        """
        Orchestrates the data loading, and timestamps' partitioning
        """
        print(f"[{self.__class__.__name__}] Cargando datos desde: {self.filepath}...")
        df = pd.read_parquet(self.filepath)
        df['VehiclePresent'] = df['VehiclePresent'].astype(float)
        df.fillna(value=np.nan, inplace=True) # Fill missing values with NaN as PyPOTS expects NaNs for imputation

        # Data splits based on timestamps
        ts_unique = np.sort(df['Timestamp'].unique())
        n_ts = len(ts_unique)
        
        train_cut = ts_unique[int(n_ts * 0.60)]
        val_cut = ts_unique[int(n_ts * 0.80)]
        
        df_train = df[df['Timestamp'] <= train_cut]
        df_val = df[(df['Timestamp'] > train_cut) & (df['Timestamp'] <= val_cut)]
        df_test = df[df['Timestamp'] > val_cut]
        
        # Generate 3D tensors
        print(f"[{self.__class__.__name__}] Generando tensores 3D (Ventana: {self.window_size} pasos)...")
        X_train = self._create_sliding_windows(df_train)
        X_val = self._create_sliding_windows(df_val)
        X_test = self._create_sliding_windows(df_test)
        
        print(f"[{self.__class__.__name__}] Particiones generadas - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        self.train_dataset = {"X": X_train}
        self.val_dataset = {"X": X_val}
        self.test_dataset = {"X": X_test}

    def get_splits(self):
        return self.train_dataset, self.val_dataset, self.test_dataset