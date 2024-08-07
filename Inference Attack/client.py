import torch
import torch.nn.functional as F
import os
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Client:
    def __init__(self, client_id, model, data_loader, epochs=5):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self):
        self.model.train()
        total_loss = 0
        for inputs, labels in self.data_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.data_loader)

