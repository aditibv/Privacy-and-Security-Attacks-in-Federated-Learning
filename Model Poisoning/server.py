import torch
import copy
import random
from neuralnet import create_model
from client import Client

class Server:
    def __init__(self, num_clients, num_rounds, client_list, val_dl, test_dl, device):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device

        # Infer the input dimension from the validation dataset
        input_dim = next(iter(val_dl))['features'].shape[1]
        self.global_model = create_model(input_dim).to(device)
        self.client_list = client_list
        self.global_dict = None

    def initialize_clients(self):
        for i in range(self.num_clients):
            client_id = f"client {i}"
            self.client_list[client_id] = Client(copy.deepcopy(self.global_model), i, self.device)
        print(self.client_list)

    def push_new_model(self):
        for client in self.client_list.values():
            client.model.load_state_dict(self.global_dict)

    def server_merge(self, nameList):
        self.global_dict = copy.deepcopy(self.global_model.state_dict())
        for layer in self.global_dict:
            summation = sum(copy.deepcopy(self.client_list[client].model.state_dict()[layer]) for client in nameList)
            self.global_dict[layer] = summation / len(nameList)

        self.global_model.load_state_dict(self.global_dict)
        self.push_new_model()

    def test(self, data_loader):
        self.global_model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in data_loader:
                features, label = batch['features'].to(self.device), batch['label'].to(self.device)
                prediction = self.global_model(features)
                total += prediction.size(0)
                correct += (prediction.argmax(1) == label).sum().item()
        return correct * 100. / total

    def degrade_model(self):
        # Introduce less noise to degrade the global model performance
        with torch.no_grad():
            for param in self.global_model.parameters():
                param.add_(torch.randn_like(param) * 0.075)  # Adjust the noise level as needed

    def run_attack_round(self, selected_clients, malicious_client_ids, round_number):
        for client_id in selected_clients:
            is_malicious = int(client_id.split()[1]) in malicious_client_ids
            if is_malicious:
                print(f"{client_id} is performing a poisoning attack.")
            self.client_list[client_id].client_training(malicious=is_malicious)
        self.server_merge(selected_clients)
        
        # Degrade the global model after merging
        self.degrade_model()
        
        # Evaluate model performance after attack
        accuracy = self.test(self.val_dl)
        return accuracy
