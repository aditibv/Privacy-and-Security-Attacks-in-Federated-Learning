import torch
from torch import nn, optim
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Client:
    def __init__(self, model, client_id, device):
        self.model = model
        self.client_id = client_id
        self.device = device
        self.train_dl = None
        self.epochs = 5  # Increased number of epochs
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        self.loss_function = nn.CrossEntropyLoss()

    def client_training(self, malicious=False):
        self.model.train()
        for e in range(self.epochs):
            total_samples = 0
            correct_samples = 0
            train_loss = 0.0

            for batch in self.train_dl:
                data, target = batch['features'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                total_samples += target.size(0)
                correct_samples += (output.argmax(1) == target).sum().item()

            train_accuracy = 100. * correct_samples / total_samples
            self.scheduler.step()
            print(f"Client {self.client_id} - Epoch {e+1} - Train Accuracy: {train_accuracy:.2f}%")

        if malicious:
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)  # Add small noise to the model
            print(f"Client {self.client_id} performed a poisoning attack.")

        return self.model.state_dict()
