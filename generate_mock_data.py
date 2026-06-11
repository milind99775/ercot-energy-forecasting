"""
ERCOT Mock Data Generator
Author: Milind Verma
Description: Generates realistic hourly synthetic data matching the schema of 
             'POC Sample Data 1.xlsx' so the forecasting pipeline can run.
"""

import os
import numpy as np
import pandas as pd

def generate_mock_dataset(output_path="data/POC Sample Data 1.xlsx"):
    print("Generating synthetic ERCOT load and price dataset...")
    
    # 1. Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 2. Define Time Range (Hourly data for ~7 months to match split_date '2022-07-02')
    date_range = pd.date_range(start="2022-01-01 00:00:00", end="2022-08-15 23:00:00", freq="H")
    n_records = len(date_range)
    
    # 3. Create Base DataFrame with Datetime columns
    df = pd.DataFrame({
        'DATETIME': date_range,
        'PEAKTYPE': [np.nan] * n_records,  # Code expects PEAKTYPE to be entirely null
        'HOURENDING': date_range.hour + 1,
        'MARKETDAY': date_range.normalize(),
        'MONTH': date_range.month,
        'YEAR': date_range.year
    })
    
    # 4. Generate load forecast columns (Range: 1000 to 5000 MW with daily cycles)
    load_columns = [
        'WZ_Coast (BIDCLOSE_LOAD_FORECAST)', 'WZ_ERCOT (BIDCLOSE_LOAD_FORECAST)',
        'WZ_East (BIDCLOSE_LOAD_FORECAST)', 'WZ_FarWest (BIDCLOSE_LOAD_FORECAST)',
        'WZ_North (BIDCLOSE_LOAD_FORECAST)', 'WZ_NorthCentral (BIDCLOSE_LOAD_FORECAST)',
        'WZ_SouthCentral (BIDCLOSE_LOAD_FORECAST)', 'WZ_Southern (BIDCLOSE_LOAD_FORECAST)',
        'WZ_West (BIDCLOSE_LOAD_FORECAST)', 'WZ_Coast (RTLOAD)', 'WZ_ERCOT (RTLOAD)',
        'WZ_East (RTLOAD)', 'WZ_FarWest (RTLOAD)', 'WZ_North (RTLOAD)',
        'WZ_NorthCentral (RTLOAD)', 'WZ_SouthCentral (RTLOAD)', 'WZ_Southern (RTLOAD)',
        'WZ_West (RTLOAD)'
    ]
    
    # Add a cyclic pattern + noise
    hour_effect = np.sin(2 * np.pi * date_range.hour / 24)
    for col in load_columns:
        base_load = np.random.uniform(1500, 4000)
        noise = np.random.normal(0, 150, n_records)
        df[col] = base_load + (base_load * 0.2 * hour_effect) + noise
        
    # 5. Generate Wind & Solar columns (Range: 0 to 1000 MW)
    renewables_cols = [
        'ERCOT (WIND_STWPF_BIDCLOSE)', 'GR_COASTAL (WIND_STWPF_BIDCLOSE)',
        'GR_ERCOT (WIND_STWPF_BIDCLOSE)', 'GR_NORTH (WIND_STWPF_BIDCLOSE)',
        'GR_PANHANDLE (WIND_STWPF_BIDCLOSE)', 'GR_SOUTH (WIND_STWPF_BIDCLOSE)',
        'GR_WEST (WIND_STWPF_BIDCLOSE)', 'NORTH (ERCOT) (WIND_STWPF_BIDCLOSE)',
        'SOUTH_HOUSTON (WIND_STWPF_BIDCLOSE)', 'WEST (ERCOT) (WIND_STWPF_BIDCLOSE)',
        'WEST_NORTH (WIND_STWPF_BIDCLOSE)', 'ERCOT (SOLAR_STPPF_BIDCLOSE)',
        'GR_COASTAL (WINDDATA)', 'GR_ERCOT (WINDDATA)', 'GR_NORTH (WINDDATA)',
        'GR_PANHANDLE (WINDDATA)', 'GR_SOUTH (WINDDATA)', 'GR_WEST (WINDDATA)',
        'ERCOT (GENERATION_SOLAR_RT)'
    ]
    for col in renewables_cols:
        df[col] = np.random.uniform(100, 800, n_records) + np.random.normal(0, 50, n_records)
        df[col] = df[col].clip(lower=0) # No negative renewable generation
        
    # 6. Generate Gas Prices (Range: $2.5 to $5.5)
    df['Katy (GASPRICE)'] = np.random.uniform(2.8, 4.5, n_records)
    df['Henry (GASPRICE)'] = df['Katy (GASPRICE)'] + np.random.normal(0, 0.1, n_records)
    df['ERCOT (TOTAL_RESOURCE_CAP_OUT)'] = np.random.uniform(5000, 8000, n_records)
    
    # 7. Generate target Price variables: Day-Ahead (DALMP) and Real-Time (RTLMP)
    # Price is modeled as a factor of demand/load + wind generation + some random price spikes
    demand_factor = df['WZ_Coast (RTLOAD)'] / 4000
    wind_factor = df['ERCOT (WIND_STWPF_BIDCLOSE)'] / 1000
    
    df['HB_NORTH (DALMP)'] = 30 + (50 * demand_factor) - (15 * wind_factor) + np.random.normal(0, 5, n_records)
    df['HB_NORTH (RTLMP)'] = df['HB_NORTH (DALMP)'] + np.random.normal(0, 12, n_records)
    
    # Introduce random extreme price spikes (very common in ERCOT)
    spike_indices = np.random.choice(n_records, size=int(n_records * 0.01), replace=False)
    df.loc[spike_indices, 'HB_NORTH (RTLMP)'] += np.random.uniform(100, 500, len(spike_indices))
    
    # 8. Introduce occasional mock null values (as your code specifically cleans and imputes nulls)
    for col in df.columns:
        if col not in ['DATETIME', 'PEAKTYPE', 'HOURENDING', 'MARKETDAY', 'MONTH', 'YEAR']:
            null_indices = np.random.choice(n_records, size=int(n_records * 0.002), replace=False)
            df.loc[null_indices, col] = np.nan
            
    # 9. Save as Excel file
    df.to_excel(output_path, index=False)
    print(f"File successfully created at: {output_path}")

if __name__ == "__main__":
    generate_mock_dataset()
