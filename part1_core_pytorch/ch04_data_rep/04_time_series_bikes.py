import datetime

import torch
import numpy as np

def date_to_float(date_str):
    if isinstance(date_str, bytes):
        date_str = date_str.decode('utf-8')
    return datetime.strptime(date_str, '%Y-%m-%d').timestamp()

def main():
    # 4.4.1 Adding a time dimension
    try:
        bikes_numpy = np.loadtxt(
            "data/p1ch4/bike-sharing-dataset/hour-fixed.csv", 
            delimiter=",", 
            skiprows=1, 
            usecols=[0] + list(range(2, 17))
        )
        bikes = torch.from_numpy(bikes_numpy)
        
        # 4.4.2 Shaping the data by time period
        # Original: (TotalHours, Features) -> Reshape to (Days, Hours, Features)
        daily_bikes = bikes.view(-1, 24, bikes.shape[1])
        
        # Transpose to (Days, Features, Hours) for model ingestion
        daily_bikes = daily_bikes.transpose(1, 2)
        
        print(f"Original Shape: {bikes.shape}")
        print(f"Reshaped Shape (NCL): {daily_bikes.shape}")
    except Exception as e:
        print(f"Skipping time series load (data missing): {e}")

if __name__ == "__main__":
    main()