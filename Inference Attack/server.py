import torch

class Server:
    def __init__(self, model, clients):
        self.model = model
        self.clients = clients

    def collect_updates(self):
        for client in self.clients:
            client.model.load_state_dict(self.model.state_dict())

    def aggregate_updates(self):
        total_params = {key: torch.zeros_like(val) for key, val in self.model.state_dict().items()}
        for client in self.clients:
            client_params = client.model.state_dict()
            for key in total_params.keys():
                total_params[key] += client_params[key]
        
        # Averaging the parameters
        for key in total_params.keys():
            total_params[key] /= len(self.clients)
        
        self.model.load_state_dict(total_params)

    def inference_attack(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            total_counts = torch.zeros(10, dtype=torch.int32)
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                for label in labels:
                    total_counts[label] += 1

        predicted_dominant_digit = torch.argmax(total_counts).item()
        actual_dominant_digits = [torch.argmax(torch.bincount(client.data_loader.dataset.tensors[1])).item() for client in self.clients]
        return predicted_dominant_digit, actual_dominant_digits
