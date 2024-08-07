import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd

BATCH_SIZE = 32

def load_data():
    # Load your preprocessed LendingClub dataset here
    file_path = r'C:\Users\MCTI Student\Documents\Final Project -PSFL\combined_loan.csv' 
    data = pd.read_csv(file_path)  # Modify this line to your dataset path
    features = torch.tensor(data.drop(columns=['loan_status']).values, dtype=torch.float32)
    targets = torch.tensor(data['loan_status'].values, dtype=torch.long)

    dataset = TensorDataset(features, targets)
    return dataset

def create_dataloaders(dataset, num_clients=10, batch_size=BATCH_SIZE):
    # Determine the lengths for each client, ensuring the sum is equal to the dataset length
    dataset_length = len(dataset)
    lengths = [dataset_length // num_clients] * num_clients
    lengths[-1] += dataset_length % num_clients  # Add the remainder to the last client

    client_datasets = random_split(dataset, lengths)
    client_dataloaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    return client_dataloaders
