# import os
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib

# def load_and_preprocess(raw_dir="data/raw"):
#     print(f"Scanning {raw_dir} for the 6 expected CSV files...")
#     file_mapping = {
#         "Benign_train.pcap.csv": 0,
#         "TCP_IP-DDoS-ICMP1_train.pcap.csv": 1,
#         "TCP_IP-DoS-ICMP1_train.pcap.csv": 2,
#         "Recon-OS_Scan_train.pcap.csv": 3,
#         "MQTT-Malformed_Data_train.pcap.csv": 4,
#         "ARP_Spoofing_train.pcap.csv": 5
#     }
    
#     if not os.path.exists(raw_dir):
#         raise FileNotFoundError(f"Directory not found: {raw_dir}")
        
#     dataframes = []
#     for expected_file in file_mapping.keys():
#         filepath = os.path.join(raw_dir, expected_file)
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"CRITICAL: Missing required file: {expected_file}")
            
#         df = pd.read_csv(filepath)
#         df['Attack_Type'] = file_mapping[expected_file]
#         dataframes.append(df)
#         print(f"Loaded {expected_file} (Initial rows: {len(df):,})")
        
#     master_df = pd.concat(dataframes, ignore_index=True)
#     initial_total = len(master_df)
    
#     cols_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Label']
#     master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns], errors='ignore')
    
#     for col in master_df.columns:
#         master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    
#     # Track NaNs/Infs separately
#     master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     before_nan = len(master_df)
#     master_df.dropna(inplace=True)
#     nan_dropped = before_nan - len(master_df)
    
#     feature_cols = [c for c in master_df.columns if c != 'Attack_Type']
    
#     # Handle Label Conflicts: Find rows with identical features but DIFFERENT labels
#     conflicts = master_df.groupby(feature_cols)['Attack_Type'].nunique()
#     conflict_indices = conflicts[conflicts > 1].index
#     if len(conflict_indices) > 0:
#         print(f"Found {len(conflict_indices):,} unique feature vectors with conflicting labels. Dropping them entirely.")
#         # Drop all rows that match these conflicting feature vectors
#         master_df = master_df.set_index(feature_cols)
#         master_df = master_df.drop(index=conflict_indices)
#         master_df = master_df.reset_index()
#     else:
#         print("No label conflicts found for identical feature vectors.")
    
#     # Now that conflicts are gone, safely drop true exact duplicates
#     before_dup = len(master_df)
#     master_df = master_df.drop_duplicates(subset=feature_cols)
#     dup_dropped = before_dup - len(master_df)
    
#     print(f"\nAudit: Dropped {nan_dropped:,} NaNs/Infs. Dropped {dup_dropped:,} exact duplicates.")
    
#     X = master_df[feature_cols].values
#     y = master_df['Attack_Type'].values
    
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
    
#     train_dataset = TensorDataset(
#         torch.tensor(X_train_scaled, dtype=torch.float32),
#         torch.tensor(y_train, dtype=torch.long)
#     )
#     val_dataset = TensorDataset(
#         torch.tensor(X_val_scaled, dtype=torch.float32),
#         torch.tensor(y_val, dtype=torch.long)
#     )
    
#     print(f"Training set: {len(X_train):,} rows, {X_train.shape[1]} features")
#     return train_dataset, val_dataset, feature_cols, scaler


import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess(raw_dir="data/raw"):
    # Map class labels based on keywords in the filename instead of exact matches
    keyword_mapping = {
        "Benign": 0,
        "TCP_IP-DDoS": 1,  # Shortened to catch both ICMP and ICMP1
        "TCP_IP-DoS": 2,   # Shortened to catch both ICMP and ICMP1
        "Recon-OS_Scan": 3,
        "MQTT-Malformed_Data": 4,
        "ARP_Spoofing": 5
    }
    
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Directory not found: {raw_dir}")
        
    # Dynamically grab ALL csv files
    all_csvs = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not all_csvs:
        raise FileNotFoundError(f"CRITICAL: No CSV files found in {raw_dir}")
        
    print(f"Scanning {raw_dir} for all CSV files (Found {len(all_csvs)})...")
    
    dataframes = []
    for file in all_csvs:
        filepath = os.path.join(raw_dir, file)
        
        # Find which keyword matches the current file
        assigned_label = None
        for keyword, label_idx in keyword_mapping.items():
            if keyword in file:
                assigned_label = label_idx
                break
                
        if assigned_label is None:
            print(f"Warning: Unrecognized file {file}, skipping.")
            continue
            
        df = pd.read_csv(filepath)
        df['Attack_Type'] = assigned_label
        dataframes.append(df)
        print(f"Loaded {file} -> Class {assigned_label} (Initial rows: {len(df):,})")
        
    master_df = pd.concat(dataframes, ignore_index=True)
    initial_total = len(master_df)
    
    cols_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Label']
    master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns], errors='ignore')
    
    for col in master_df.columns:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    
    # Track NaNs/Infs separately
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before_nan = len(master_df)
    master_df.dropna(inplace=True)
    nan_dropped = before_nan - len(master_df)
    
    feature_cols = [c for c in master_df.columns if c != 'Attack_Type']
    
    # Handle Label Conflicts: Find rows with identical features but DIFFERENT labels
    conflicts = master_df.groupby(feature_cols)['Attack_Type'].nunique()
    conflict_indices = conflicts[conflicts > 1].index
    if len(conflict_indices) > 0:
        print(f"Found {len(conflict_indices):,} unique feature vectors with conflicting labels. Dropping them entirely.")
        # Drop all rows that match these conflicting feature vectors
        master_df = master_df.set_index(feature_cols)
        master_df = master_df.drop(index=conflict_indices)
        master_df = master_df.reset_index()
    else:
        print("No label conflicts found for identical feature vectors.")
    
    # Now that conflicts are gone, safely drop true exact duplicates
    before_dup = len(master_df)
    master_df = master_df.drop_duplicates(subset=feature_cols)
    dup_dropped = before_dup - len(master_df)
    
    print(f"\nAudit: Dropped {nan_dropped:,} NaNs/Infs. Dropped {dup_dropped:,} exact duplicates.")
    
    X = master_df[feature_cols].values
    y = master_df['Attack_Type'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    print(f"Training set: {len(X_train):,} rows, {X_train.shape[1]} features")
    return train_dataset, val_dataset, feature_cols, scaler