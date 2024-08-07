import torch
from torch.utils.data import DataLoader, random_split
from neuralnet import SimpleNN
from dataloader import load_data, create_dataloaders
from server import aggregate_models
from client import Client
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

def main():
    dataset = load_data()
    client_dataloaders = create_dataloaders(dataset)
    
    input_dim = dataset[0][0].shape[0]
    output_dim = len(torch.unique(torch.tensor([y for _, y in dataset])))

    global_model = SimpleNN(input_dim, output_dim).to(device)
    
    num_clients = len(client_dataloaders)
    num_malicious_clients = int(0.3 * num_clients)
    clients = []

    for i in range(num_clients):
        is_malicious = i < num_malicious_clients
        client_dataset_length = len(client_dataloaders[i].dataset)
        train_length = int(0.8 * client_dataset_length)
        test_length = client_dataset_length - train_length
        train_dataset, test_dataset = random_split(client_dataloaders[i].dataset, [train_length, test_length])
        
        train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        client = Client(global_model, i, train_dl, test_dl, is_malicious)
        clients.append(client)
    
    all_train_accuracies = []
    all_val_accuracies = []
    for r in range(5):
        client_models = []
        round_train_accuracies = []
        round_val_accuracies = []
        for client in clients:
            client_state_dict, train_accuracies, val_accuracies = client.client_training()
            client_models.append(client_state_dict)
            round_train_accuracies.append(train_accuracies)
            round_val_accuracies.append(val_accuracies)
        
        global_model = aggregate_models(global_model, client_models)
        all_train_accuracies.append(round_train_accuracies)
        all_val_accuracies.append(round_val_accuracies)
        print(f"Round {r+1} completed.")

    torch.save(global_model.state_dict(), 'global_model.pth')
    
    plot_accuracies(all_train_accuracies, all_val_accuracies, num_clients)

def plot_accuracies(train_accuracies, val_accuracies, num_clients):
    malicious_train_acc = []
    malicious_val_acc = []
    non_malicious_train_acc = []
    non_malicious_val_acc = []

    for r in range(len(train_accuracies)):
        for c in range(num_clients):
            if c < int(0.3 * num_clients):
                malicious_train_acc.append(train_accuracies[r][c])
                malicious_val_acc.append(val_accuracies[r][c])
            else:
                non_malicious_train_acc.append(train_accuracies[r][c])
                non_malicious_val_acc.append(val_accuracies[r][c])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(malicious_train_acc)), [sum(x)/len(x) for x in malicious_train_acc], label='Malicious Clients Train Accuracy', color='red')
    plt.plot(range(len(non_malicious_train_acc)), [sum(x)/len(x) for x in non_malicious_train_acc], label='Non-Malicious Clients Train Accuracy', color='green')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(malicious_val_acc)), [sum(x)/len(x) for x in malicious_val_acc], label='Malicious Clients Val Accuracy', color='red')
    plt.plot(range(len(non_malicious_val_acc)), [sum(x)/len(x) for x in non_malicious_val_acc], label='Non-Malicious Clients Val Accuracy', color='green')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
