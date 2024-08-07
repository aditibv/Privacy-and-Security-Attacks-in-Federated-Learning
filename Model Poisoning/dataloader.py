import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler

class LendingClubDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop(columns=['loan_status']).values  # Replace 'loan_status' with your actual label column name
        self.labels = self.data['loan_status'].values

        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'features': torch.tensor(self.features[idx], dtype=torch.float32),
                  'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        return sample

def load_data(filepath, num_clients):
    dataset = LendingClubDataset(filepath)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    BATCH_SIZE = 32

    train_loaders = []
    for i in range(num_clients):
        client_size = len(train_dataset) // num_clients
        client_data, train_dataset = random_split(train_dataset, [client_size, len(train_dataset) - client_size])
        train_loaders.append(DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True))

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loaders, val_loader, test_loader
