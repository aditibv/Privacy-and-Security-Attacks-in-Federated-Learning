import torch
from torch.utils.data import DataLoader, random_split
from client import Client
from neuralnet import SimpleNN
from dataloader import load_data, create_dataloaders
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

def aggregate_models(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i][k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def main():
    dataset = load_data()
    client_dataloaders = create_dataloaders(dataset)
    input_dim = dataset[0][0].shape[0]
    output_dim = len(torch.unique(torch.tensor([y for _, y in dataset])))

    global_model = SimpleNN(input_dim, output_dim).to(device)
    
    num_clients = len(client_dataloaders)
    num_malicious_clients = int(0.3 * num_clients)  # 30% of clients are malicious
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
    
    history = {
        'train_acc_malicious': [],
        'val_acc_malicious': [],
        'train_acc_non_malicious': [],
        'val_acc_non_malicious': []
    }
    
    for r in range(5):  # number of rounds
        client_models = []
        round_train_acc_malicious = []
        round_val_acc_malicious = []
        round_train_acc_non_malicious = []
        round_val_acc_non_malicious = []
        
        for client in clients:
            client_state_dict, train_acc, val_acc = client.client_training()
            client_models.append(client_state_dict)
            if client.is_malicious:
                round_train_acc_malicious.append(train_acc)
                round_val_acc_malicious.append(val_acc)
            else:
                round_train_acc_non_malicious.append(train_acc)
                round_val_acc_non_malicious.append(val_acc)
        
        global_model = aggregate_models(global_model, client_models)
        print(f"Round {r+1} completed.")
        
        history['train_acc_malicious'].append(sum(round_train_acc_malicious) / len(round_train_acc_malicious))
        history['val_acc_malicious'].append(sum(round_val_acc_malicious) / len(round_val_acc_malicious))
        history['train_acc_non_malicious'].append(sum(round_train_acc_non_malicious) / len(round_train_acc_non_malicious))
        history['val_acc_non_malicious'].append(sum(round_val_acc_non_malicious) / len(round_val_acc_non_malicious))
    
    torch.save(global_model.state_dict(), 'global_model.pth')
    
    # Plotting the results and saving them
    rounds = range(1, 6)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(rounds, history['train_acc_malicious'], label='Malicious Clients Train Accuracy')
    plt.plot(rounds, history['train_acc_non_malicious'], label='Non-Malicious Clients Train Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Rounds')
    plt.legend()
    plt.savefig('training_accuracy_over_rounds.png')
    
    plt.subplot(1, 2, 2)
    plt.plot(rounds, history['val_acc_malicious'], label='Malicious Clients Validation Accuracy')
    plt.plot(rounds, history['val_acc_non_malicious'], label='Non-Malicious Clients Validation Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Rounds')
    plt.legend()
    plt.savefig('validation_accuracy_over_rounds.png')
    
    plt.tight_layout()
    plt.close()

if __name__ == "__main__":
    main()
