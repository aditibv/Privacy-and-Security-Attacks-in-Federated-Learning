import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from dataloader import load_data
from server import Server

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Specify the filepath to your preprocessed Lending Club data
    filepath = r'C:\Users\MCTI Student\Documents\Final Project -PSFL\Final Project\processed_loan_data\combined_loan.csv'
    
    # Load preprocessed data
    num_clients = 30
    num_malicious_clients = 5  # Number of malicious clients
    train_dls, val_dl, test_dl = load_data(filepath, num_clients)

    num_rounds = 5

    client_list = {}
    for i in range(num_clients):
        client_list[f"client {i}"] = None

    serv = Server(num_clients, num_rounds, client_list, val_dl, test_dl, device)
    serv.initialize_clients()

    # Mark some clients as malicious
    malicious_client_ids = random.sample(range(num_clients), num_malicious_clients)
    
    for i, client in enumerate(serv.client_list.values()):
        client.train_dl = train_dls[i]

    attack_results = []
    for round in range(num_rounds):
        print(f"ROUND: {round + 1}/{num_rounds}")
        C = random.random()
        num_selected = max(int(num_clients * C), 1)
        client_index = np.random.permutation(num_clients)[:num_selected]
        print(f"Selected clients: {client_index}")

        selected_clients = []
        for c in client_index:
            selected_clients.append(f"client {c}")

        accuracy_after_attack = serv.run_attack_round(selected_clients, malicious_client_ids, round)
        
        if accuracy_after_attack is not None:
            print(f"Accuracy after attack in round {round + 1}: {accuracy_after_attack:.2f}%")
            attack_results.append(accuracy_after_attack)
        else:
            print(f"Error: Failed to get accuracy after attack in round {round + 1}")

    # Plot results
    rounds = range(1, num_rounds + 1)
    plt.plot(rounds, attack_results, marker='o', label='Overall Accuracy')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Effect of Model Poisoning Attack on Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("attack_results.png")
    plt.show()

if __name__ == "__main__":
    main()
