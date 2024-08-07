# Privacy-and-Security-Attacks-in-Federated-Learning

## Introduction

Federated Learning (FL) is revolutionizing machine learning by enabling collaborative model training across distributed devices while preserving data privacy through local data storage and centralized model update sharing. Despite its benefits, this decentralized approach introduces security risks from malicious participants. This study investigates the vulnerabilities of FL systems, including data poisoning, model poisoning, and inference attacks. By simulating these attacks, their impact on model performance and data integrity is evaluated, and robust defense strategies, including advanced aggregation techniques and differential privacy methods, are proposed to enhance the security and reliability of FL systems.

###  Implementations

- **Data Poisoning Attack:** Involves injecting malicious data into client datasets to degrade model performance.
- **Model Poisoning Attack:** Clients manipulate model updates to affect global model performance.
- **Inference Attack:** Adversaries analyze model outputs to infer sensitive information about training data.

## Methodology

### Data Collection and Preprocessing

- **Data Collection:** Utilizes Lending Club Loan Data https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans, a comprehensive financial dataset.
- **Preprocessing:** Cleans and normalizes data, encodes categorical features, and splits the dataset into local subsets for federated learning simulation.

### Federated Learning Setup

- **Environment Simulation:** Uses PyTorch to build and train models, simulating FL with multiple local datasets and a central server.
- **Client-Side Implementation:** Clients train models locally and send updates to the server.
- **Server-Side Aggregation:** Aggregates client updates to form a global model.

### Implementation of Attacks

- **Data Poisoning:** Introduces mislabeled data and noise to manipulate the global model.
- **Model Poisoning:** Adds noise to model parameters during training to degrade performance.
- **Inference Attacks:** Uses shadow models to infer membership in the training dataset.

### Evaluation

The effectiveness of attacks and defenses is assessed through:

- Performance metrics comparison between different attacks.
- Graphical and statistical analyses to highlight impacts and vulnerabilities.

## Conclusion

This research provides practical insights and recommendations to enhance the security and reliability of Federated Learning systems. By simulating various attacks and  proposing defense strategies, the study aims to improve FL's robustness against potential threats.


