import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os

# Avoid OpenMP runtime issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
lf = nn.CrossEntropyLoss()

class Client:
    def __init__(self, model, client_id, train_dl, test_dl, is_malicious=False):
        self.model = model
        self.client_id = client_id
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.epochs = 1
        self.optimizer = optim.SGD(self.model.parameters(), lr)
        self.is_malicious = is_malicious
        if self.is_malicious:
            self.train_dl = self.poison_data(train_dl)

    def poison_data(self, train_dl):
        poisoned_data = []
        for data, target in train_dl.dataset:
            if torch.rand(1).item() < 0.1:  # Poison 10% of the data
                target = torch.randint(0, 2, target.shape).long()  # Randomize the labels (example)
            poisoned_data.append((data, target))
        return DataLoader(poisoned_data, batch_size=train_dl.batch_size, shuffle=True)

    def client_training(self):
        self.model.train()
        train_accuracies = []
        val_accuracies = []
        for e in range(self.epochs):
            ttl = 0
            crct = 0
            trainAccuracy = 0.0
            loss = 0
            prediction = None
            for batch_index, (data, target) in enumerate(self.train_dl):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                prediction = self.model(data)
                ttl += prediction.size(0)
                loss = lf(prediction, target)
                loss.backward()
                self.optimizer.step()
                crct += (prediction.argmax(1) == target).sum()
            trainAccuracy = crct * 100. / ttl
            train_accuracies.append(trainAccuracy.item())
            
            with torch.no_grad():
                self.model.eval()
                total = 0
                correct = 0
                validationAccuracy = 0
                prediction2 = None
                for i, (x, y) in enumerate(self.test_dl):
                    (x, y) = (x.to(device), y.to(device))
                    prediction2 = self.model(x)
                    total += prediction2.size(0)
                    correct += (prediction2.argmax(1) == y).sum()
                validationAccuracy = correct * 100. / total
                val_accuracies.append(validationAccuracy.item())
                
            print(f"EPOCH {e + 1}/{self.epochs} SUMMARY FOR CLIENT {self.client_id}:")
            print(f"Malicious Client: {self.is_malicious}")
            print(f"Validation Accuracy: {validationAccuracy}, and Train Accuracy is: {trainAccuracy}")
            print("==============================================")
        
        return self.model.state_dict(), train_accuracies, val_accuracies
