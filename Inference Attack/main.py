from server import Server
from client import Client
from dataloader import get_data_loaders
from neuralnet import SimpleNN
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def plot_losses(client_losses, num_rounds, num_epochs):
    for client_id, losses in client_losses.items():
        plt.plot(range(1, num_rounds * num_epochs + 1), losses, label=f'Client {client_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss per Client')
    plt.legend()
    plt.show()

def main():
    # Define the number of clients and epochs
    num_clients = 5
    num_epochs = 2  # Reduced number of epochs

    # Load preprocessed data
    train_loaders, test_loader = get_data_loaders("C:\\Users\\MCTI Student\\Documents\\Final Project -PSFL\\combined_loan.csv", num_clients)
    print("Data Loaded Successfully")

    # Input size is inferred from the first batch
    input_size = next(iter(train_loaders[0]))[0].shape[1]
    print(f"Input size: {input_size}")

    # Initialize model, clients, and server
    model = SimpleNN(input_size)
    clients = [Client(client_id=i, model=model, data_loader=train_loader, epochs=num_epochs) for i, train_loader in enumerate(train_loaders)]
    server = Server(model, clients)

    num_rounds = 5
    client_losses = {i: [] for i in range(len(clients))}

    # Training
    for round_num in range(1, num_rounds + 1):
        print(f'Round {round_num}/{num_rounds}')
        server.collect_updates()
        server.aggregate_updates()

        for client_id, client in enumerate(clients):
            for epoch in range(client.epochs):
                train_loss = client.train()
                client_losses[client_id].append(train_loss)
                print(f'Client {client_id}, Epoch {epoch + 1}/{client.epochs}, Train Loss: {train_loss:.4f}')

    # Perform inference attack
    predicted_dominant_digit, actual_dominant_digits = server.inference_attack(test_loader)
    print(f'Predicted dominant digit in the clients\' dataset: {predicted_dominant_digit}')
    for client_id, actual_dominant_digit in enumerate(actual_dominant_digits):
        print(f'Client {client_id} actual dominant digit: {actual_dominant_digit}')
        if predicted_dominant_digit == actual_dominant_digit:
            print(f'Prediction matches for client {client_id}.')
        else:
            print(f'Prediction does not match for client {client_id}.')

    # Plot training losses
    plot_losses(client_losses, num_rounds, clients[0].epochs)

if __name__ == '__main__':
    main()
