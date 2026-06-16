import duckdb
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# DuckDB raw data
root_dir = os.environ.get("PROJECT_ROOT", "")
db_file = os.environ.get("DDBB_PATH", "duck_db/occupancy.duckdb")
db_path = os.path.join(root_dir, db_file)

# Export processed parquet
data_folder = os.path.join("DATA_FOLDER_PATH_IPYNB", "../../data/")
df_model_name = os.environ.get("PARQUET_API_MODEL", "df_pypots_api.parquet")
output_path = os.path.join(data_folder, df_model_name)

conn = duckdb.connect(db_path, read_only=True)
query = """
    SELECT 
        kerbsideid AS DeviceId,
        CAST(status_timestamp AS TIMESTAMP) AS Timestamp,
        status_description,
        location.lon AS lon,
        location.lat AS lat
    FROM occupancy
    WHERE status_timestamp IS NOT NULL
"""
df = conn.execute(query).fetchdf()
conn.close()
print("Muestra del dataset inicial crudo:")
print(df.head())
print(f"\nDimensiones iniciales del dataset: {df.shape}")

df['VehiclePresent'] = df['status_description'].apply(lambda x: 1.0 if x == 'Present' else 0.0) # map target to binary values
df.drop(columns=['status_description'], inplace=True)

# Avoid old sensors returned by the API, only get the data from the established date range
START_DATE = os.environ.get("START_DATE","2026-05-01 00:00:00")
END_DATE = os.environ.get("END_DATE","2026-05-31 23:59:59")
df_time = df[df['Timestamp'] <= END_DATE].copy() # only filter by end_date to keep data prior to the start date for forward fill

# Remove duplicates and sort
df_time = df_time.drop_duplicates(subset=['DeviceId', 'Timestamp'], keep='last')
df_time = df_time.sort_values(by=['DeviceId', 'Timestamp'])

print(df_time.shape)

# Build 15 min grid
grid_freq = os.environ.get("GRID_FREQ", "15min")
time_grid = pd.date_range(start=START_DATE, end=END_DATE, freq=grid_freq) # filter by the established date range
unique_devices = df_time['DeviceId'].unique()
full_index = pd.MultiIndex.from_product([unique_devices, time_grid], names=['DeviceId', 'Timestamp'])

# Apply forward fill in next time freq if there is no state update
df_indexed = df_time.set_index(['DeviceId', 'Timestamp'])
df_continuous = df_indexed.reindex(full_index, method='ffill').reset_index()
print(df_continuous.shape)

# Apply same trigonometric transformations as in the historic dataset
df_continuous['Hour'] = df_continuous['Timestamp'].dt.hour + df_continuous['Timestamp'].dt.minute / 60.0
df_continuous['Hour_sin'] = np.sin(2 * np.pi * df_continuous['Hour'] / 24.0)
df_continuous['Hour_cos'] = np.cos(2 * np.pi * df_continuous['Hour'] / 24.0)

df_continuous['WeekDay'] = df_continuous['Timestamp'].dt.dayofweek
df_continuous['WeekDay_sin'] = np.sin(2 * np.pi * df_continuous['WeekDay'] / 7.0)
df_continuous['WeekDay_cos'] = np.cos(2 * np.pi * df_continuous['WeekDay'] / 7.0)

df_continuous['Month'] = df_continuous['Timestamp'].dt.month
df_continuous['Month_sin'] = np.sin(2 * np.pi * df_continuous['Month'] / 12.0)
df_continuous['Month_cos'] = np.cos(2 * np.pi * df_continuous['Month'] / 12.0)

df_continuous.drop(columns=['Hour', 'WeekDay', 'Month'], inplace=True)
print(df_continuous.shape)

# Reorder columns to match the historic dataframe structure 
column_order = ['DeviceId', 'Timestamp', 'VehiclePresent', 'lon', 'lat', 
                'Hour_sin', 'Hour_cos', 'WeekDay_sin', 'WeekDay_cos', 'Month_sin', 'Month_cos']
df_final = df_continuous[column_order]
print("\nMuestra del dataset final procesado:")
print(f"\nDimensiones finales del dataset: {df_final.shape}")
print(df_final.head())
print(df_final.tail())

# Export data
# df_final.to_parquet(output_path, index=False)
# print(f"Dataframe exportado a {output_path}")