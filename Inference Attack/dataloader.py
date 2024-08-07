import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def get_data_loaders(file_path, num_clients, batch_size=32):
    # Load preprocessed data
    data = pd.read_csv(file_path)
    print("Data Loaded Successfully")
    
    # Assuming the target column is 'loan_status' and the features are all other columns
    target_column = 'loan_status'
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Split train data among clients
    train_loaders = []
    client_data_size = len(X_train) // num_clients
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = (i + 1) * client_data_size
        train_dataset = TensorDataset(X_train[start_idx:end_idx], y_train[start_idx:end_idx])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    
    # Create test loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loaders, test_loader
